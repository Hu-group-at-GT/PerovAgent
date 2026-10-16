from __future__ import annotations

import math
import sys
import warnings

from ase.calculators.mixing import SumCalculator
from ase.optimize import LBFGS

from perovagent.evaluation.construction import ConstructionTools, infer_bonds, ligand_graph_intact

_RELAXER = None


class Relaxer:
    """The measurement: MatterSim + D3 + LBFGS. Relaxes structures and returns energies."""

    def __init__(self, construction, cfg):
        """cfg: a perovagent.config.RelaxerConfig (device, checkpoint, d3, optimizer)."""
        from mattersim.forcefield import Potential, MatterSimCalculator
        from perovagent.evaluation.dftd3 import DFTD3Calculator
        warnings.filterwarnings("ignore", message="No PBC detected")

        self.construction = construction
        self.fmax = cfg.optimizer.fmax
        self.max_steps = cfg.optimizer.max_steps

        print("[Evaluation] Loading MatterSim + NVIDIA D3 calculator...")
        mattersim = Potential.from_checkpoint(load_path=cfg.checkpoint, device=cfg.device)
        mattersim_calc = MatterSimCalculator(potential=mattersim, device=cfg.device)
        d3 = DFTD3Calculator(a1=cfg.d3.a1, a2=cfg.d3.a2, s8=cfg.d3.s8, cutoff=cfg.d3.cutoff,
                             device=cfg.device)
        self.calculator = SumCalculator([d3, mattersim_calc])

    def optimize(self, atoms):
        """Run LBFGS on atoms to cfg.optimizer's fmax/max_steps. Returns (energy, converged)."""
        atoms.calc = self.calculator
        opt = LBFGS(atoms, logfile=None)
        converged = opt.run(self.fmax, steps=self.max_steps)
        return atoms.get_potential_energy(), converged

    def relax_full(self, smiles, motif, seed, save_dir, name):
        """Relax the covered slab with the anionic ligand placed."""
        slab = self.construction.build_covered_slab(smiles, motif, name, seed)
        if slab is None:
            return None, None
        try:
            E, _ = self.optimize(slab.atoms)
        except Exception as e:
            print(f"[Evaluation] relax_full failed: {e}")
            return None, None
        self.construction.save_slab(slab, save_dir, name)

        ok, why = ligand_graph_intact(slab, smiles)
        if not ok:
            print(f"[Evaluation] relax_full: ligand REACTED on the surface -- {why}; dropping "
                  f"this placement (its E_bind would book the reaction energy as binding)")
            return None, {"error": "ligand_reacted_on_surface"}
        try:
            contact_map = self.construction.contact_map(slab)
        except Exception as e:
            print(f"[Evaluation] contact-map failed: {e}")
            contact_map = {}
        return float(E), contact_map

    def relax_vacancy(self, smiles, motif, seed, save_dir, name):
        """E of the relaxed slab with one anionic ligand removed from the SAME
        unrelaxed placement (E_vacancy)."""
        slab = self.construction.build_covered_slab(smiles, motif, name, seed)
        if slab is None:
            return None
        try:
            slab.remove_ligand(id=slab.ligands[-1].id)   # last placed = anionic
            E, _ = self.optimize(slab.atoms)
        except Exception as e:
            print(f"[Evaluation] relax_vacancy failed: {e}")
            return None
        self.construction.save_slab(slab, save_dir, name)
        return float(E)

    def relax_ligand(self, smiles, seed, save_dir, name, shard):
        """One shard of the E_ligand conformer search."""
        built = self.construction.build_ligand_conformers(smiles, seed)
        if not built:
            raise RuntimeError("ligand conformer ensemble build failed; refusing to "
                               "score a truncated search")
        confs, ref_bonds = built

        n_total = len(confs)
        k, n = shard
        confs = confs[k::n]
        if not confs:
            return (None, None, 0)   # more shards than conformers

        results = []  # (energy, converged, atoms)
        n_isomerised = 0
        for i, cand in enumerate(confs):
            try:
                E, converged = self.optimize(cand)
                E = float(E)
            except Exception as e:  # noqa: BLE001
                print(f"[Evaluation] relax_ligand conformer {i} failed: {e}")
                continue
            if not math.isfinite(E):
                print(f"[Evaluation] relax_ligand conformer {i} returned non-finite energy")
                continue
            if infer_bonds(cand) != ref_bonds:
                n_isomerised += 1
                continue
            results.append((E, converged, cand))
        if n_isomerised:
            print(f"[Evaluation] relax_ligand: discarded {n_isomerised}/{len(confs)} conformers "
                  f"that rearranged during relaxation")
        if not results:
            print(f"[Evaluation] relax_ligand: no conformer in this shard produced a finite "
                  f"energy with the input connectivity intact "
                  f"({len(confs)} tried of {n_total} shortlisted)")
            return (None, None, n_isomerised)

        converged = [r for r in results if r[1]]
        e_any = min(r[0] for r in results)
        best = min(converged or results, key=lambda r: r[0])   # per-shard winner
        e_conv = best[0] if converged else None
        self.construction.save_ligand(best[2], save_dir, name)
        return (e_conv, e_any, n_isomerised)


def build_relaxer(construction, relaxer, contact_cutoff: float):
    """Construct the MatterSim+D3 Relaxer (with its ConstructionTools) from a validated
    ConstructionConfig + RelaxerConfig. Called per worker; the main process never builds tools."""
    return Relaxer(construction=ConstructionTools(construction, contact_cutoff), cfg=relaxer)


# worker helpers
def _worker_init(configs):
    """Build this worker's own Relaxer (loads MatterSim once)."""
    global _RELAXER
    construction, relaxer, contact_cutoff = configs
    _RELAXER = build_relaxer(construction, relaxer, contact_cutoff)


def _preflight_build(configs):
    construction, relaxer, contact_cutoff = configs
    try:
        build_relaxer(construction, relaxer, contact_cutoff)
        return (True, "")
    except Exception as e:  # noqa: BLE001
        return (False, f"{type(e).__name__}: {e}")


def _job(args):
    """One unit of relaxation work."""
    mol_idx, kind, seed, smiles, motif, save_dir, name, extra = args
    motif = list(motif)
    try:
        if kind == "full":
            E, contact_map = _RELAXER.relax_full(smiles, motif, seed, save_dir, name)
            return (mol_idx, kind, seed, E, contact_map)
        if kind == "vac":
            return (mol_idx, kind, seed,
                    _RELAXER.relax_vacancy(smiles, motif, seed, save_dir, name), None)
        e_conv, e_any, n_iso = _RELAXER.relax_ligand(smiles, seed, save_dir, name,
                                                     extra)   # "lig"
        return (mol_idx, kind, seed, e_conv, {"e_any": e_any, "n_isomerised": n_iso})
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"mol{mol_idx} {kind} seed{seed} failed: {e}\n")
        return (mol_idx, kind, seed, None, None)
