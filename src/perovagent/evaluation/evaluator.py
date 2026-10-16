from __future__ import annotations

import atexit
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from perovagent.chemistry.anchors import AnchorRegistry, to_bound_form
from perovagent.chemistry.admission import AdmittedLigand
from perovagent.evaluation.construction import _check_test_ligand_is_placed, b_donor_cutoff
from perovagent.evaluation.relaxation import _job, _preflight_build, _worker_init

log = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    descriptors: dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _anchor_reaches_b_site(contact_map: dict, b_site: str) -> bool:
    """True unless some placed ligand's anchor donor fails to contact a `b_site` atom within its
    cutoff. `b_site` is the configured B-site element (construction.B). A ligand with no donor in
    the map cannot be judged and passes."""
    for entry in contact_map.values():
        anchors = [h for h in entry if h["is_donor_atom"]]
        if not anchors:
            continue
        if not any(c["element"] == b_site and c["distance"] <= b_donor_cutoff(b_site, h["element"])
                   for h in anchors for c in h["contacts"]):
            return False                  # this ligand's anchor never reached the B-site
    return True


def _aggregate(mol, full, vac, lig, contact_maps, seeds, full_err):
    """Combine per-placement energies into the binding-energy descriptor."""
    def _finite(x) -> bool:
        return x is not None and math.isfinite(x)

    be = []
    n_nonfinite = 0
    for s in seeds:
        f, v = full.get(s), vac.get(s)
        if f is None or v is None or lig is None:
            continue
        if not (_finite(f) and _finite(v)):
            n_nonfinite += 1
            continue
        be.append(f - v - lig)
    n_reacted = len(full_err)
    if not be:
        if n_reacted:
            err = "ligand_reacted_on_surface"
        else:
            err = "non_finite_energies" if n_nonfinite else "all_placements_failed"
        return {**mol, "binding_energy": None, "error": err}
    mean = statistics.mean(be)
    # contact map from the most stable (lowest-E_full) placement
    fe = {s: e for s, e in full.items() if _finite(e)}
    contact_map = contact_maps.get(min(fe, key=fe.get))
    return {
        **mol,
        "binding_energy": mean,
        "binding_energy_std": statistics.pstdev(be),
        "n_placements_ok": len(be),
        "n_placements_reacted": n_reacted,
        "contact_map": contact_map or {},
    }


def _mol_key(m: dict) -> str:
    """Stable 10-char id for one (molecule, anchor) evaluation, used as its save-dir name."""
    key = f"{m['label']}::{m['anchor']}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def _reduce_ligand_shards(conv, any_, n_ran, n_expected, n_iso, mol_idx):
    """E_ligand from one molecule's shard results, or None to drop the molecule."""
    if n_ran < n_expected:
        log.warning("mol%d: ligand conformer search INCOMPLETE (%d/%d shards ran); dropping the "
                    "molecule rather than scoring a truncated search", mol_idx, n_ran, n_expected)
        return None
    if n_iso:
        log.info("mol%d: discarded %d ligand conformer(s) that rearranged during relaxation",
                 mol_idx, n_iso)
    if conv:
        return min(conv)
    if any_:
        e = min(any_)
        log.warning("mol%d: no ligand conformer converged; using the lowest unconverged "
                    "(%.4f eV)", mol_idx, e)
        return e
    return None


def _log_batch_timing(save_root, *, n_molecules, n_jobs, n_workers,
                      t_start, t_first, t_end, n_results, pool_warm) -> None:
    """One JSON line per evaluated batch, appended to save_root/batch_timing.jsonl."""
    rec = {"n_molecules": n_molecules, "n_jobs": n_jobs, "n_workers": n_workers,
           "pool_warm": pool_warm,
           "startup_s": None if t_first is None else round(t_first - t_start, 3),
           "relaxation_s": None if t_first is None else round(t_end - t_first, 3),
           "total_s": round(t_end - t_start, 3), "t": time.time(),
           "n_results": n_results, "aborted": n_results < n_jobs}
    try:
        with open(os.path.join(save_root, "batch_timing.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError as e:
        log.warning("could not write batch timing: %s", e)


def _new_worker_pool(construction, relaxer, contact_cutoff, n_workers):
    """A spawn-context Pool of exactly `n_workers` workers, each with its own MatterSim loaded
    by `_worker_init`. All of them share the single GPU this job was allocated."""
    ctx = mp.get_context("spawn")
    return ctx.Pool(max(1, n_workers), initializer=_worker_init,
                    initargs=((construction, relaxer, contact_cutoff),))


def compute_many(mols, construction, n_placements, seed_base, n_workers, save_root, e_vac_ref,
                 pool, pool_warm):
    prepared, jobs = [], []
    lig_expected = {}          # mol_idx -> how many lig shards were dispatched (completeness check)
    seeds = list(range(seed_base, seed_base + n_placements))
    # Conformer-search shards: one per worker, capped at n_ligand_relax, floored at 1.
    n_lig_shards = max(1, min(n_workers, construction.n_ligand_relax))
    for i, m in enumerate(mols):
        spec = m["anchor_spec"]
        bound = to_bound_form(m["smiles"], spec)
        if bound is None:
            prepared.append({**m, "binding_energy": None, "error": "bound_form_or_motif_failed"})
            continue
        motif = tuple(spec.coord_atoms)
        prepared.append({**m, "_idx": i})
        # Content-addressed save dir, keyed on (molecule, anchor), not the batch slot.
        sd = os.path.join(save_root, "mol_" + _mol_key(m))
        os.makedirs(sd, exist_ok=True)
        try:
            with open(os.path.join(sd, "meta.json"), "w") as fh:
                json.dump({"smiles": m["smiles"], "anchor": m["anchor"],
                           "label": m["label"], "bound_smiles": bound}, fh, indent=2)
        except OSError:
            pass
        for s in seeds:
            jobs.append((i, "full", s, bound, motif, os.path.join(sd, f"full_s{s}"),
                         f"full_s{s}", None))
            # Vacancy slab is a constant; with a cached reference these jobs are not created.
            if e_vac_ref is None:
                jobs.append((i, "vac", s, bound, motif,
                             os.path.join(sd, f"vac_s{s}"), f"vac_s{s}", None))
        for k in range(n_lig_shards):
            jobs.append((i, "lig", seed_base, bound, motif, sd, f"ligand_{k}",
                         (k, n_lig_shards)))
        lig_expected[i] = n_lig_shards

    raw = []          # hoisted: a batch that dies before dispatch must still log
    if jobs:
        t_start = time.perf_counter()
        t_first = None
        try:
            it = pool.imap_unordered(_job, jobs)
            for _ in range(len(jobs)):
                try:
                    raw.append(it.next(timeout=1800))     # 30 min for one relaxation
                except mp.TimeoutError:
                    raise mp.TimeoutError(
                        "a relaxation returned no result within 1800s -- a worker "
                        "is hung or failing to initialize (bad checkpoint / device); aborting "
                        "this batch")
                if t_first is None:
                    t_first = time.perf_counter()
        finally:
            _log_batch_timing(save_root, n_molecules=len(lig_expected), n_jobs=len(jobs),
                              n_workers=max(1, n_workers), t_start=t_start, t_first=t_first,
                              t_end=time.perf_counter(), n_results=len(raw),
                              pool_warm=pool_warm)

    full = {i: {} for i in range(len(mols))}

    vac = ({i: {s: e_vac_ref for s in seeds} for i in range(len(mols))}
           if e_vac_ref is not None else {i: {} for i in range(len(mols))})

    # Per-molecule conformer results, accumulated across shards.
    lig_conv = {i: [] for i in range(len(mols))}     # mins over CONVERGED intact conformers
    lig_any = {i: [] for i in range(len(mols))}      # mins ignoring convergence
    lig_iso = {i: 0 for i in range(len(mols))}
    lig_ran = {i: 0 for i in range(len(mols))}       # shards that ran to completion
    contact_maps = {i: {} for i in range(len(mols))}
    full_err = {i: {} for i in range(len(mols))}    # seed -> failure reason from relax_full
    for mol_idx, kind, seed, E, payload in raw:
        if kind == "full":
            full[mol_idx][seed] = E
            if payload:
                if "error" in payload:
                    full_err[mol_idx][seed] = payload["error"]
                else:
                    contact_maps[mol_idx][seed] = payload
        elif kind == "vac":
            vac[mol_idx][seed] = E
        elif payload is not None:          # "lig": a shard that ran to completion
            lig_ran[mol_idx] += 1
            if E is not None:
                lig_conv[mol_idx].append(E)
            if payload["e_any"] is not None:
                lig_any[mol_idx].append(payload["e_any"])
            lig_iso[mol_idx] += payload["n_isomerised"]

    # Resolve E_ligand across shards
    lig = {i: _reduce_ligand_shards(lig_conv[i], lig_any[i], lig_ran[i], n, lig_iso[i], i)
           for i, n in lig_expected.items()}

    out = []
    for p in prepared:
        p.pop("anchor_spec")
        if "_idx" not in p:
            out.append(p); continue
        i = p.pop("_idx")
        out.append(_aggregate(p, full[i], vac[i], lig[i], contact_maps[i], seeds, full_err[i]))
    return out


class BindingEnergyEvaluator:
    """Scores an admitted ligand by its surface binding energy.
    Descriptor: E_bind = E_full - E_vacancy - E_ligand, averaged over `n_placements`."""

    def __init__(self, save_root: str, *, construction, relaxer, n_placements: int,
                 placement_seed_base: int, n_workers: int, max_placement_std: float,
                 min_placements_ok: int, max_binding_energy: float, e_vac_ref: Optional[float],
                 contact_cutoff: float, registry: AnchorRegistry):
        self.save_root = save_root
        os.makedirs(self.save_root, exist_ok=True)
        self.max_placement_std = max_placement_std
        self.min_placements_ok = min_placements_ok
        self.n_placements = n_placements
        self.placement_seed_base = placement_seed_base
        self.e_vac_ref = e_vac_ref
        self.max_binding_energy = max_binding_energy
        self.construction = construction
        self.relaxer = relaxer
        self.contact_cutoff = contact_cutoff
        self.n_workers = n_workers
        self.registry = registry
        self._pool = None

    def _worker_pool(self):
        """The campaign-long worker pool, built on first use."""
        if self._pool is None:
            self._pool = _new_worker_pool(
                self.construction, self.relaxer, self.contact_cutoff, self.n_workers)
            atexit.register(self.close)
            log.info("[Evaluation] worker pool up: %d workers sharing one GPU, one MatterSim "
                     "load each. Every later batch reuses them, so a further load line in the "
                     "log means a worker was respawned.", max(1, self.n_workers))
        return self._pool

    def close(self) -> None:
        """Tear down the worker pool."""
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.terminate()
            pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def preflight(self) -> tuple[bool, str]:
        ctx = mp.get_context("spawn")
        try:
            with ctx.Pool(1) as pool:
                ok, reason = pool.map_async(
                    _preflight_build, [(self.construction, self.relaxer, self.contact_cutoff)]
                ).get(timeout=600)[0]
        except mp.TimeoutError:
            return False, ("evaluator did not initialize within 600s -- the MatterSim "
                           "load or the GPU appears to be hanging")
        except Exception as e:  # noqa: BLE001
            return False, f"evaluator preflight crashed: {type(e).__name__}: {e}"
        return ok, reason

    def evaluate_batch(self, ligands: list[AdmittedLigand]) -> dict:
        """Evaluate many admitted ligands in one parallel batch, keyed by canonical SMILES."""
        mols = [{"label": lig.canonical_smiles, "smiles": lig.canonical_smiles,
                 "anchor": lig.anchor_type, "anchor_spec": self.registry.get(lig.anchor_type)}
                for lig in ligands]
        warm = self._pool is not None
        results = compute_many(
            mols, self.construction,
            n_placements=self.n_placements, seed_base=self.placement_seed_base,
            n_workers=self.n_workers, save_root=self.save_root,
            e_vac_ref=self.e_vac_ref, pool=self._worker_pool(), pool_warm=warm)
        return {lig.canonical_smiles: self._binding_result(r)
                for lig, r in zip(ligands, results)}

    def _binding_result(self, r: dict) -> EvaluationResult:
        be = r["binding_energy"]
        if be is None:
            return EvaluationResult(failure_reason=r["error"])
        if be < self.max_binding_energy:
            return EvaluationResult(failure_reason=f"unphysical_binding_energy:{be:.2f}")
        n_ok = r["n_placements_ok"]
        if n_ok < self.min_placements_ok:
            n_reacted = r["n_placements_reacted"]
            reason = (f"ligand_reacted_on_surface:{n_reacted}/{self.n_placements}_placements"
                      if n_reacted else
                      f"too_few_placements:{n_ok}/{self.n_placements}")
            return EvaluationResult(failure_reason=reason)
        # GUARD 1 -- placement spread: if independent placements of the same molecule disagree
        # beyond the threshold, the pose is undefined and the mean is not a measurement.
        std = r["binding_energy_std"]
        if self.max_placement_std > 0 and std > self.max_placement_std:
            return EvaluationResult(failure_reason=f"unstable_placement:std={std:.2f}eV")
        # GUARD 2 -- X-site coordination: if the anchor donor never reaches a B-site cation, the
        # ligand is adsorbed on the surface (A-site contacts / dispersion), not passivating the
        # X-site.
        if not _anchor_reaches_b_site(r["contact_map"], self.construction.B):
            return EvaluationResult(failure_reason="anchor_not_coordinating_bsite")
        return EvaluationResult(
            descriptors={"binding_energy": be, "binding_energy_std": std},
            metadata={"n_placements_ok": n_ok,
                      "n_placements_reacted": r["n_placements_reacted"],
                      "ligand_contact_map": r["contact_map"]},
        )


def build_binding_evaluator(
    save_root: str,
    construction,
    relaxer,
    binding,
    n_workers: int,
    *,
    placement_seed_base: int,
    contact_cutoff: float,
    registry: AnchorRegistry,
) -> BindingEnergyEvaluator:
    """Construct a BindingEnergyEvaluator (binding-energy descriptor)."""
    # e_vac_ref None => relax the vacancy per placement; a number => use binding.e_vacancy_cached
    # as E_vacancy for every placement.
    e_vac_ref = None
    if binding.use_reference_cache:
        anionic = list((construction.binding_sites or {}).get("anionic", []))
        if len(anionic) != 1:
            raise ValueError(
                f"binding.use_reference_cache requires exactly one anionic binding site, got "
                f"{anionic}. relax_vacancy removes only the LAST ligand, so with several sites "
                f"E_vacancy is 'slab with N-1 ligands still adsorbed' -- molecule-dependent.")
        e_vac_ref = binding.e_vacancy_cached
        log.info("[Evaluation] E_vacancy from binding.e_vacancy_cached = %.4f eV "
                 "(per-placement vacancy relaxations disabled)", e_vac_ref)
    return BindingEnergyEvaluator(
        save_root=save_root,
        n_placements=binding.n_placements,
        max_placement_std=binding.max_placement_std,
        min_placements_ok=binding.min_placements_ok,
        max_binding_energy=binding.max_binding_energy,
        e_vac_ref=e_vac_ref,
        placement_seed_base=placement_seed_base,
        construction=construction,
        relaxer=relaxer,
        n_workers=n_workers,
        contact_cutoff=contact_cutoff,
        registry=registry,
    )


def build_evaluator(config, *, output_dir: str, n_workers: int, seed: int,
                    registry: AnchorRegistry) -> BindingEnergyEvaluator:
    """Build the binding-energy evaluator from a DiscoveryConfig + the run-level values."""
    _check_test_ligand_is_placed(config.construction)
    donors = set(config.admission.allowed_atom_symbols) - {"C", "H"}
    if not donors:
        raise ValueError("admission.allowed_atom_symbols leaves no element but C to act as an "
                         "anchor donor")
    contact_cutoff = max(b_donor_cutoff(config.construction.B, x) for x in donors)
    log.info("[Evaluation] contact-map radius %.3f A (longest %s--donor cutoff over %s)",
             contact_cutoff, config.construction.B, sorted(donors))
    return build_binding_evaluator(
        save_root=f"{output_dir}/perovkit_eval",
        construction=config.construction,
        relaxer=config.relaxer,
        binding=config.binding,
        placement_seed_base=100 * seed,
        n_workers=n_workers,
        contact_cutoff=contact_cutoff,
        registry=registry,
    )
