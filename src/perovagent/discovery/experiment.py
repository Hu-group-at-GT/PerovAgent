from __future__ import annotations

import logging
from typing import Optional

from rdkit import Chem

from perovagent.discovery.principles import (DesignPrinciple, satisfies, predicate_text)
from perovagent.discovery.blackboard import CandidateRecord
from perovagent.chemistry.anchors import AnchorRegistry
from perovagent.config import DiscoveryConfig
from perovagent.evaluation.construction import describe_slab_system, format_contact_map
from perovagent.evaluation.evaluator import build_evaluator
from perovagent.chemistry.admission import (admit_ligand, describe_admission_constraints,
                                            molecule_chemistry_gate)
from perovagent.agents.proposer import LLMProposer, Proposal

log = logging.getLogger(__name__)


ANCHOR_EDIT_WARNING = (
    "Before reusing an anchor name, check the edited molecule against its registered `motif` "
    "SMARTS. A non-matching child is rejected before evaluation. An edit to the head itself -- "
    "swapping one of its atoms, inserting an atom between head and scaffold, or removing one of "
    "its ionizable atoms -- requires a new lowercase anchor name and `anchor_spec`. A substituent "
    "edit can also break a motif that names substituent atoms: a motif that pins the substituents "
    "on the head atom stops matching once any one of them changes. "
    "A registered motif is improvised from whichever molecule first carried it, so it is "
    "routinely narrower than its name suggests and can reach out of the head into atoms you "
    "think of as scaffold. Read the parent's `motif` line atom by atom and ask, for every "
    "atom and bond it names, whether your child still supplies it. Only an edit landing "
    "entirely outside the matched motif keeps the parent's anchor name; anything else needs "
    "an `anchor_spec` whose `motif_smarts` covers the head as you actually built it.")


class ExperimentRunner:
    def __init__(self, config: DiscoveryConfig, *, output_dir: str, n_workers: int, seed: int):
        self.config = config
        self.registry = AnchorRegistry()
        self.evaluator = build_evaluator(
            config, output_dir=output_dir, n_workers=n_workers, seed=seed, registry=self.registry)
        # the slab system and the admission gates; the decider's system prompt carries it too
        self.system_context = (describe_slab_system(config.construction)
                               + "\n\n" + describe_admission_constraints(config.admission))
        self.proposer = LLMProposer(model=config.campaign.model,
                                    system_context=self.system_context,
                                    effort=config.campaign.reasoning_effort,
                                    material=config.construction.material,
                                    b_site=config.construction.B,
                                    max_binding_energy=config.binding.max_binding_energy,
                                    effect_size_threshold=config.campaign.effect_size_threshold)
        self._seen: set[str] = set()
        self._anchor_pool: dict[str, list[float]] = {}
        self.failed: list[tuple] = []
        self.rejects: list[tuple] = []
        self.duplicates: list[tuple] = []
        self.rejected_log: list[dict] = []
        self.iteration: int = 0
        self.evals_charged: int = 0

    def _record_rejected(self, kind: str, smiles: str, reason: str, *, proposal: Proposal,
                         source: str, anchor: str = "") -> None:
        self.rejected_log.append({
            "iter": self.iteration, "kind": kind, "smiles": smiles, "reason": reason,
            "anchor": anchor, "source": source,
            "hypothesis": proposal.hypothesis,
            "rationale": proposal.rationale,
            "parent_smiles": proposal.parent_smiles or "",
            "predicted_direction": proposal.predicted_direction,
        })

    def _admit(self, smiles: str, declared: str):
        """Run a proposal through the admission gate for its declared anchor."""
        cfg = self.config
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return None, "", "smiles_parse_failed"
        reason = molecule_chemistry_gate(mol, cfg.admission)
        if reason is not None:
            return None, "", reason
        spec = self.registry.get(declared)
        ligand, r = admit_ligand(mol, spec, cfg.admission)
        anchor_used = declared
        if ligand is None:
            for other in self.registry.specs():
                if other.name == declared or other.binding_mode != spec.binding_mode:
                    continue
                cand, _ = admit_ligand(mol, other, cfg.admission)
                if cand is not None:
                    log.warning("anchor fallback %s -> %s for %s (declared failed: %s)",
                                declared, other.name, smiles, r)
                    ligand, anchor_used = cand, other.name
                    break
            else:
                return None, "", (f"anchor_mismatch:{declared}: your molecule does not "
                                  f"contain the registered motif {spec.motif_smarts!r} "
                                  f"({r})")
        if ligand.canonical_smiles in self._seen:
            return ligand, anchor_used, "duplicate"
        return ligand, anchor_used, ""

    def _ensure_anchor(self, anchor: str, spec, smiles: str):
        anchor = anchor.strip()
        if anchor in self.registry:
            return anchor, None
        if spec is None:
            return None, f"unknown_anchor:{anchor} (no anchor_spec supplied to register it)"
        if spec.name != anchor:
            spec = spec.model_copy(update={"name": anchor})
        ok, reason = self.registry.register(
            spec, probe_smiles=smiles,
            allowed_elements=self.config.admission.allowed_atom_symbols)
        return (anchor, None) if ok else (None, f"anchor_spec_rejected:{reason}")

    @staticmethod
    def _measurement(r) -> tuple[Optional[dict], str]:
        """The measured fields of one EvaluationResult, or (None, its failure reason)."""
        if r.failure_reason:
            return None, r.failure_reason
        return {"binding_energy": r.descriptors["binding_energy"],
                "binding_energy_std": r.descriptors["binding_energy_std"],
                "n_placements_ok": r.metadata["n_placements_ok"],
                "n_placements_reacted": r.metadata["n_placements_reacted"],
                "geometry": format_contact_map(r.metadata["ligand_contact_map"])}, ""

    def _evaluate_batch(self, ligands) -> list:
        if not ligands:
            return []
        try:
            res = self.evaluator.evaluate_batch(ligands)
        except Exception as e:
            log.warning("batch eval failed: %s", e)
            return [(None, f"batch_eval_raised:{type(e).__name__}") for _ in ligands]
        return [self._measurement(res[ligand.canonical_smiles]) for ligand in ligands]

    def evaluate_many(self, proposals: list[Proposal], source: str) -> list[CandidateRecord]:
        admitted, batch_seen = [], set()
        failed_smiles = {s for s, _ in self.failed}
        for p in proposals:
            smiles = p.smiles
            if not smiles:
                continue
            smiles = smiles.strip()
            declared, reason = self._ensure_anchor(p.anchor, p.anchor_spec, smiles)
            if declared is None:
                self.rejects.append((smiles, reason))
                self._record_rejected("rejected", smiles, reason, source=source, proposal=p)
                continue
            ligand, anchor_used, reason = self._admit(smiles, declared)
            if reason == "duplicate":
                canon = ligand.canonical_smiles
                if canon not in failed_smiles:
                    self.duplicates.append((canon, reason))
                self._record_rejected("duplicate", canon, reason, anchor=anchor_used,
                                      source=source, proposal=p)
                continue
            if ligand is None:
                self.rejects.append((smiles, reason))
                self._record_rejected("rejected", smiles, reason, anchor=declared,
                                      source=source, proposal=p)
                continue
            if ligand.canonical_smiles in batch_seen:
                self.duplicates.append((ligand.canonical_smiles, "duplicate_within_batch"))
                self._record_rejected("duplicate", ligand.canonical_smiles,
                                      "duplicate_within_batch", anchor=anchor_used,
                                      source=source, proposal=p)
                continue
            batch_seen.add(ligand.canonical_smiles)
            admitted.append((ligand, anchor_used, declared, p))
        results = self._evaluate_batch([ligand for ligand, _, _, _ in admitted])
        self.evals_charged += len(admitted)
        out = []
        for (ligand, anchor_used, declared, p), (measured, why) in zip(admitted, results):
            self._seen.add(ligand.canonical_smiles)
            if measured is None:
                self.failed.append((ligand.canonical_smiles, why))
                self._record_rejected("failed", ligand.canonical_smiles, why, anchor=anchor_used,
                                      source=source, proposal=p)
                continue
            self._anchor_pool.setdefault(anchor_used, []).append(measured["binding_energy"])
            out.append(CandidateRecord(
                smiles=ligand.canonical_smiles, anchor=anchor_used, declared_anchor=declared,
                source=source, **measured,
                rationale=p.rationale, hypothesis=p.hypothesis,
                predicted_direction=p.predicted_direction, parent_smiles=p.parent_smiles or ""))
        return out

    def _anchor_menu(self) -> str:
        specs = self.registry.specs()
        if not specs:
            return "No anchors are registered; supply an `anchor_spec` for every proposal."
        recent = {s.name for s in specs[-10:]}     # the 10 most recently registered
        shown = [s for s in specs if self._anchor_pool.get(s.name) or s.name in recent]
        lines = ["These are the exact registered definitions. Use a name only when the molecule "
                 "contains its motif; otherwise the proposal is rejected before evaluation. "
                 "Supply `anchor_spec` only for a new anchor:"]
        for s in shown:
            es = self._anchor_pool.get(s.name) or []
            cov = f"measured n={len(es)}, strongest {min(es):+.2f} eV" if es else "never measured"
            row = (f"  - {s.name} ({s.binding_mode}; {cov})\n"
                   f"      motif {s.motif_smarts}   donors {','.join(s.coord_atoms)}")
            if s.binding_mode == "anionic":
                row += (f"\n      deprot {s.deprot_smarts}   "
                        f"charge_target_index {s.charge_target_index}")
            lines.append(row)
        hidden = [s.name for s in specs if s not in shown]
        if hidden:
            lines.append(f"  (+{len(hidden)} more registered but never measured, definitions "
                         f"not shown: {', '.join(hidden)})")
        return "\n".join(lines)

    def propose_families(self, brief: str, n: int, *, goal: str, control: str,
                         principle_context: str, evidence: str,
                         playbook: str) -> list[CandidateRecord]:
        ctx = (principle_context + "\n\n") if principle_context else ""
        ev = (evidence + "\n\n") if evidence else ""
        playbook_block = (playbook + "\n\n") if playbook else ""
        if goal == "exploit" and control == "matched":
            mode = (
                "Refine by matched single edits. Each child must differ from one named strong "
                "parent by exactly one substituent, heteroatom, ring, or anchor edit. Preserve "
                "everything else and set `parent_smiles` to that exact parent so the energy "
                "difference is attributable to the edit.\n" + ANCHOR_EDIT_WARNING)
        elif goal == "exploit":
            mode = (
                "Refine the named strongest binders with close analogs and small single edits. "
                "Set `parent_smiles` to the analog being modified.\n" + ANCHOR_EDIT_WARNING)
        else:  # explore
            mode = (
                "Explore broadly across scaffold families (monocyclic, fused-aromatic, "
                "bridged/cage, spiro, saturated-ring, heteroaromatic, and flexible/acyclic) and "
                "anchor chemistries without assuming a preferred class.")
        out: list[CandidateRecord] = []
        for rnd in range(max(1, self.config.campaign.max_retry)):
            need = n - len(out)
            if need <= 0:
                break
            prompt = self._experiment_prompt(brief, need, mode, ctx, ev, playbook_block)
            resp = self.proposer.propose(prompt)
            got = self.evaluate_many(resp.proposals, source=f"experiment:{goal}")
            out.extend(got)
            if not got and rnd:
                break
        return out

    def _rejected_block(self) -> str:
        retry = ""
        if self.failed:
            lines = "\n".join(f"  - {s} : {r}" for s, r in self.failed[-8:])
            retry = (
                f"Evaluation failures ({len(self.failed)} molecules; GPU budget spent). "
                f"Do not repeat these candidates. Recent reasons:\n{lines}\n"
                f"Reason guide:\n"
                f"- `too_few_placements`: the ligand could not be seated at the X-site; reduce "
                f"steric difficulty.\n"
                f"- `anchor_not_coordinating_bsite`: the ligand was seated, but the donor did "
                f"not contact {self.config.construction.B}; revise the head or donor reach.\n"
                f"- `unphysical_binding_energy`: an invalid, overly negative relaxation, not "
                f"a superb binder.\n"
                f"- `unstable_placement`: a pose-dependent result.\n"
                f"- `ligand_reacted_on_surface`: intraligand connectivity changed, invalidating "
                f"the energy. Related analogs remain possible, but repeated failure warrants "
                f"changing the implicated feature or leaving the series. Weak binding itself "
                f"still returns a valid number.\n\n")
        if self.rejects:
            lines = "\n".join(f"  - {s} : {r}" for s, r in self.rejects[-8:])
            retry += (
                f"Admission rejections ({len(self.rejects)} molecules; no GPU budget spent). "
                f"Recent reasons:\n{lines}\n"
                f"Correct the stated gate failure. `anchor_spec_rejected:` refers to the supplied "
                f"specification, not necessarily the molecule. `satisfies Nx ... (need 1)` means "
                f"the motif is too broad; describe the complete head so it matches once.\n\n")
        if self.duplicates:
            shown, uniq = set(), []
            for s, r in reversed(self.duplicates):
                if s in shown:
                    continue
                shown.add(s)
                uniq.append((s, r))
                if len(uniq) >= 8:
                    break
            uniq.reverse()
            n_distinct = len({s for s, _ in self.duplicates})
            lines = "\n".join(
                f"  - {s}" + ("   (repeated within one response)"
                              if r == "duplicate_within_batch" else "")
                for s, r in uniq)
            retry += (
                f"Exact duplicates ({len(self.duplicates)} proposals covering {n_distinct} "
                f"molecules). "
                f"They produced no measurement or GPU charge but occupied proposal slots. Recent:\n"
                f"{lines}\n"
                f"This is bookkeeping, not chemical failure: a repeat is no evidence against "
                f"the molecule. The evidence sample is not the complete history; consult the "
                f"roster below. New close analogs remain valid. Treat the most obvious single "
                f"edits of a previously refined parent as already attempted: change a different "
                f"position, or use a different substituent class, on that same parent -- or move "
                f"to a different parent. Do not abandon the scaffold or parent that the round "
                f"brief fixes for you.\n\n")
        return retry + self._tried_roster()

    def _tried_roster(self) -> str:
        """The complete do-not-repeat set."""
        if not self._seen:
            return ""
        smiles = sorted(self._seen)
        rows = "\n".join("  " + "  ".join(smiles[i:i + 4]) for i in range(0, len(smiles), 4))
        return (
            f"Complete do-not-repeat roster ({len(smiles)} measured or failed molecules). Exact "
            f"matches are rejected; molecules absent from this list and new close analogs remain "
            f"eligible.\n{rows}\n\n")

    def _experiment_prompt(self, brief: str, n: int, mode: str, ctx: str, ev: str,
                           playbook_block: str) -> str:
        contract = (
            f"=== This round ===\n"
            f"Objective: find X-site passivants for pristine "
            f"{self.config.construction.material} with progressively stronger (more negative) "
            f"surface binding energy.\n"
            f"Deliver: exactly {n} valid X-site-passivating ligands.\n"
            f"Mode: {mode}\n"
            f"Brief: {brief}\n\n"
            f"Use the output and anchor definitions in the system instructions; the mode and "
            f"brief above provide the round-specific constraints.\n\n"
            f"=== Registered anchors ===\n{self._anchor_menu()}\n\n")
        return (contract
                + ev
                + playbook_block
                + ctx
                + self._rejected_block()
                + f"Now produce the {n} ligands specified under This round above.")

    @staticmethod
    def _fit_groups(ask_m: int, ask_o: int, room: int) -> tuple[int, int]:
        if ask_m + ask_o <= room:
            return ask_m, ask_o
        half = room // 2
        if ask_m <= half:
            return ask_m, room - ask_m
        if ask_o <= half:
            return room - ask_o, ask_o
        return half, room - half

    def propose_for_predicate(self, principle: DesignPrinciple, n: int, *, matched: bool,
                              principle_context: str, evidence: str,
                              playbook: str) -> list[CandidateRecord]:
        min_per_group = self.config.campaign.prospective_min_per_group
        recs: list[CandidateRecord] = []
        n_match = n_other = 0
        budget = max(2 * min_per_group, n) + min_per_group
        charged0 = self.evals_charged
        prev_failed: list = []
        context = "\n\n".join(block for block in (evidence, playbook, principle_context) if block)
        for rnd in range(self.config.campaign.max_retry):
            need_m, need_o = max(0, min_per_group - n_match), max(0, min_per_group - n_other)
            if need_m == 0 and need_o == 0:
                break
            room = budget - (self.evals_charged - charged0)
            if room <= 0:
                log.info(
                    "explain: stopping at the round's charge budget %d (groups %d/%d, need >=%d "
                    "each); the round will report insufficient rather than overspend", budget,
                    n_match, n_other, min_per_group)
                break
            if rnd == 0:
                ask_m, ask_o = max(need_m, n // 2), max(need_o, n - n // 2)
            else:
                ask_m, ask_o = (need_m + 1 if need_m else 0), (need_o + 1 if need_o else 0)
            ask_m, ask_o = self._fit_groups(ask_m, ask_o, room)
            prompt = self._predicate_experiment_prompt(
                principle, ask_m=ask_m, ask_o=ask_o,
                have_m=n_match, have_o=n_other, min_per_group=min_per_group, rnd=rnd,
                matched=matched, group_failures=prev_failed)
            if context:
                prompt += "\n\n" + context
            prompt += (f"\n\nNow produce exactly {ask_m} satisfying and {ask_o} violating "
                       f"ligands as specified under This round above.")
            failed_mark = len(self.failed)
            resp = self.proposer.propose(prompt)
            for rec in self.evaluate_many(resp.proposals, source="experiment:explain"):
                recs.append(rec)
                if satisfies(principle.predicate, rec.smiles):
                    n_match += 1
                else:
                    n_other += 1
            prev_failed = self.failed[failed_mark:]
        return recs

    def _group_failure_block(self, principle, group_failures) -> str:
        if not group_failures:
            return ""
        groups: dict[str, list] = {"satisfying": [], "violating": []}
        for smi, reason in group_failures:
            side = "satisfying" if satisfies(principle.predicate, smi) else "violating"
            groups[side].append((smi, reason))
        parts = []
        for side, rows in groups.items():
            if not rows:
                continue
            listed = "\n".join(f"  - {s} : {r}" for s, r in rows[:6])
            more = f"\n  (+{len(rows) - 6} more)" if len(rows) > 6 else ""
            plural = "molecule" if len(rows) == 1 else "molecules"
            parts.append(f"{len(rows)} {side} {plural} reached the GPU and failed:\n"
                         f"{listed}{more}")
        lopsided = ""
        n_sat, n_vio = len(groups["satisfying"]), len(groups["violating"])
        if not min(n_sat, n_vio) and max(n_sat, n_vio) >= 3:
            side = "satisfying" if n_sat else "violating"
            lopsided = (
                f" Every failure was on the {side} side, so it is that arm's chemotype that "
                f"will not seat, not the predicate: change the scaffold or head that keeps "
                f"failing rather than proposing more variants of it.")
        return ("Last round's attrition, by arm." + lopsided
                + "\n" + "\n".join(parts) + "\n\n")

    def _predicate_experiment_prompt(self, principle, ask_m, ask_o, have_m, have_o,
                                     min_per_group, rnd, matched: bool,
                                     group_failures: list) -> str:
        claim = ("bind more strongly (more negative E_bind)"
                 if principle.predicted_direction == "lower" else
                 "bind more weakly (less negative E_bind)")
        need_m, need_o = max(0, min_per_group - have_m), max(0, min_per_group - have_o)
        if ask_m == 0 or ask_o == 0:
            want, other = (("violate", ask_o) if ask_m == 0 else ("satisfy", ask_m))
            why = ("The other arm already has enough molecules; additional proposals there do "
                   "not improve this test."
                   if ((need_m == 0) if ask_m == 0 else (need_o == 0)) else
                   "The remaining budget covers only one arm in this round.")
            design = (
                f"This is a one-arm top-up. {why} Propose exactly {other} distinct valid X-site "
                f"ligands that {want} the predicate, and no molecules for the other arm. "
                + ("Keep them closely matched to the molecules already measured (similar "
                   "scaffold/size) so the contrast stays attributable to the predicate. "
                   if matched else ""))
        elif matched:
            design = (
                f"Propose matched pairs: each satisfying ligand has a near-identical violating "
                f"twin differing only in the predicate feature. Keep the same anchor, scaffold, "
                f"and size wherever the predicate permits. If the predicate is the anchor head, "
                f"the anchor must differ; then match scaffold and size as closely as possible. "
                f"Link each pair through `parent_smiles` -- name the satisfying molecule on its "
                f"violating twin -- so the two are scored as a matched pair rather than as two "
                f"independent groups. "
                f"Aim for ~{max(ask_m, ask_o)} pairs ({ask_m} satisfying + "
                f"{ask_o} violating).")
        else:
            design = (
                f"Propose {ask_m} distinct valid X-site ligands that satisfy the predicate and "
                f"{ask_o} that violate it. Keep size and anchor similar so the predicate remains "
                f"the main difference.")
        design += (
            " A violating molecule must not match the predicate. Change the anchor head when the "
            "predicate itself defines that head; a slightly less-matched violator is preferable "
            "to an unbalanced set that cannot test the contrast.\n" + ANCHOR_EDIT_WARNING)
        status = ""
        if rnd > 0:
            short = ("the violating arm" if need_o and not need_m else
                     "the satisfying arm" if need_m and not need_o else "both arms")
            status = (
                f"Status: {have_m} satisfying and {have_o} violating molecules have passed "
                f"validation. A prospective verdict needs >= {min_per_group} in each arm; "
                f"otherwise the principle remains unchanged even though successful measurements "
                f"enter the candidate pool. {short} is short (need {need_m} more satisfying, "
                f"{need_o} more violating). Ensure satisfying molecules truly match the predicate, "
                f"violating molecules truly do not, and avoid earlier proposals.\n\n")
            status += self._group_failure_block(principle, group_failures)
        return (
            f"=== This round: controlled test of one design principle ===\n"
            f"  \"{principle.statement}\"\n"
            f"Predicate: a molecule {predicate_text(principle.predicate)}.\n"
            f"Claim: molecules satisfying the predicate {claim}.\n"
            f"{design}\n\n"
            f"{status}"
            f"=== Registered anchors ===\n{self._anchor_menu()}\n"
            f"(or propose a new one with `anchor_spec`)\n\n"
            f"{self._rejected_block()}"
        )
