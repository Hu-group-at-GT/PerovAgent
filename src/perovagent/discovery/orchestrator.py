from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from statistics import median
from typing import Optional

from perovagent.chemistry.admission import canonical_identity
from perovagent.discovery.blackboard import (OBJECTIVE_LABEL, Blackboard, binding_energy_stats,
                                             fmt_binding_energy, rank_key)
from perovagent.discovery.audit import audit_principle
from perovagent.discovery.principles import (
    DesignPrinciple,
    format_supported_principles,
    prospective_verdict,
    retrospective_verdict,
    _fmt,
)
from perovagent.agents.decider import Decision
from perovagent.agents.summarizer import format_playbook

_VALIDATOR_OWNED = ("status", "effect_size", "p_value", "validation", "detail",
                    "n_match", "n_other", "n_match_eff", "n_other_eff",
                    "n_pairs", "effect_sd",
                    "retrospective_effect_size", "retrospective_p_value", "retrospective_status",
                    "within_anchor_effect", "n_usable_anchor_strata", "within_size_effect",
                    "frac_match",
                    "anchor_perfectly_collinear", "degenerate_partition", "audit_notes")

log = logging.getLogger(__name__)


def _fmt_opt(x, prec: int = 2) -> str:
    return f"{x:+.{prec}f}"


def _clip(s: str, n: int) -> str:
    s = s.replace("|", "/").replace("\n", " ").strip()
    return (s[:n] + "…") if len(s) > n else s


class DiscoveryOrchestrator:
    def __init__(self, experiment, decider, blackboard: Blackboard, log_dir: str, summarizer, *,
                 patience: int, effect_size_threshold: float,
                 retrospective_min_per_group: int, prospective_min_per_group: int,
                 min_anchor_stratum: int, significance_alpha: float, random_seed: int):
        self.experiment = experiment
        self.decider = decider
        self.effect_size_threshold = effect_size_threshold
        self.retrospective_min_per_group = retrospective_min_per_group
        self.prospective_min_per_group = prospective_min_per_group
        self.min_anchor_stratum = min_anchor_stratum
        self.significance_alpha = significance_alpha
        self.random_seed = random_seed
        self.patience = patience
        self._best_loss = None
        self._no_improve = 0
        self._no_decision = 0
        self._stop_reason = None
        self.summarizer = summarizer
        self._last_summary = None
        self._summary_n_decided = 0
        self._last_batch_recs = None
        self.bb = blackboard
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def _observe(self, playbook: str) -> str:
        bb = self.bb
        lines = [
            f"Goal: {bb.goal}",
            f"Objective: {OBJECTIVE_LABEL}. Resolution policy: |ΔE_bind| < "
            f"{self.effect_size_threshold:g} eV is treated as unresolved. "
            f"This is a decision threshold, not an uncertainty assigned to every value; where a "
            f"molecule carries a `+-sd`, that number is shown separately and is that molecule's "
            f"own placement spread.",
            f"Budget: {bb.budget.remaining_evals()} evaluations left; "
            f"iter {bb.budget.iters_spent}/{bb.budget.iters_cap}.",
        ]
        if self.patience > 0 and len(bb.pool):
            lines.append(
                f"Progress: no improvement beyond the {self.effect_size_threshold:.2f} eV "
                f"threshold for {self._no_improve}/{self.patience} iterations. The run stops "
                f"automatically at {self.patience}; choose `STOP` if repeated exploit rounds "
                f"show no improvement.")
        lines.append(
            f"Pool: {len(bb.pool)} molecules evaluated; "
            f"strongest so far = {_fmt(bb.pool.best_value())} eV.")
        att = self._rejected_summary()
        if att:
            lines.append(att)
        lines += self._coverage_lines(bb.pool.coverage())
        conc = self._concentration_line()
        if conc:
            lines.append(conc)
        lines += self._molecule_evidence_lines()
        lines += self._ledger_lines()
        lines += self._recent_action_lines()
        lines.append(self._nudge_line())
        if playbook:
            lines.append(
                "\nUse the playbook below when choosing the next experiment. Do not keep "
                "briefing edit families it marks `noisy/neutral` or lists under `Avoid:` -- that "
                "mark is a prohibition, not one consideration among several. Weigh its prediction "
                "calibration when you decide to exploit, pivot, or stop.\n" + playbook)
        lines.append("\nChoose one action now.")
        return "\n".join(lines)

    def _save_observation(self, iteration: int, obs: str) -> None:
        try:
            os.makedirs(os.path.join(self.log_dir, "observations"), exist_ok=True)
            self._atomic_write(os.path.join("observations", f"iter_{iteration:03d}.txt"),
                               lambda fh: fh.write(obs))
        except Exception as e:  # noqa: BLE001 -- an audit artifact must never break the run
            log.warning("could not persist observation for iter %d: %s: %s",
                        iteration, type(e).__name__, e)

    def _concentration_line(self) -> Optional[str]:
        """Anchor concentration panel: how much of the pool one anchor now holds."""
        recs = self.bb.pool.all()
        if not recs:
            return None
        counts: dict[str, int] = {}
        for r in recs:
            counts[r.anchor] = counts.get(r.anchor, 0) + 1
        top, n = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        share = n / len(recs)
        if n < 20 or share < 0.40:
            return None
        return (
            f"Anchor concentration: {n}/{len(recs)} molecules ({share:.0%}) carry `{top}`.\n"
            f"- To estimate a within-anchor effect, choose a feature that varies within `{top}` "
            f"and can provide >={self.min_anchor_stratum} molecules on each side.\n"
            f"- `DERIVE_PRINCIPLE` uses no evaluations.")

    def _coverage_lines(self, cov) -> list:
        """Scaffold coverage panel: top-10 scaffold families by strongest BE."""
        out = [f"Scaffold coverage: {len(cov)} families (family: count, strongest BE):"]
        for k, (c, b) in list(cov.items())[:10]:
            out.append(f"  {k}: n={c}, best={_fmt(b)}")
        return out

    def _spread(self, r) -> str:
        """Per-molecule uncertainty: the spread over independent placements of the SAME
        molecule, and how many of them survived. Both are recorded on every record and were
        being rendered nowhere, so both agents read 2-decimal energies as exact.

        A molecule that decomposed in a MINORITY of its placements is still accepted and scored
        on the survivors, and the placement spread is computed over those survivors too -- so the
        spread cannot reveal the reaction that was filtered out. Say it in the row instead."""
        bits = [f"+-{r.binding_energy_std:.2f}",
                f"{r.n_placements_ok}/{self.experiment.evaluator.n_placements} placements"]
        if r.n_placements_reacted:
            bits.append(f"{r.n_placements_reacted} reacted and were dropped; this energy is an "
                        f"average over the placements where the ligand stayed intact")
        return f" ({', '.join(bits)})"

    def _balanced_evidence(self):
        """Return the records and a de-duplicated strongest / mid-pack / weakest sample."""
        records = sorted(self.bb.pool.all(), key=rank_key)
        choices = [
            ("Strongest binders (most negative first)", records[:5]),
        ]
        mid_start = max(0, (len(records) - 3) // 2)
        # "mid-pack", not "controls": these are a rank slice of what happened to be measured,
        # not a designed baseline, and naming them controls invites reading them as one.
        choices.append(("Mid-pack (pool median region; neither best nor worst)",
                        records[mid_start:mid_start + 3]))
        choices.append(("Weakest valid binders (counterexamples; weakest first)",
                        list(reversed(records[-3:]))))
        shown, sections = set(), []
        for title, candidates in choices:
            rows = [r for r in candidates if r.smiles not in shown]
            if rows:
                shown.update(r.smiles for r in rows)
                sections.append((title, rows))
        return records, sections

    def _anchor_evidence_lines(self, records) -> list[str]:
        by_anchor = {}
        for r in records:
            by_anchor.setdefault(r.anchor, []).append(r.binding_energy)
        anchor_rows = sorted(by_anchor.items(), key=lambda kv: (min(kv[1]), kv[0]))
        out = ["Per-anchor outcomes (n, median, strongest; strongest anchor first):"]
        for name, values in anchor_rows[:12]:
            out.append(f"  {name}: n={len(values)}, median={median(values):+.2f}, "
                       f"best={min(values):+.2f}")
        if len(anchor_rows) > 12:
            out.append(f"  (+{len(anchor_rows) - 12} anchors omitted by "
                       f"the evidence-summary cap)")
        return out

    def _molecule_evidence_lines(self) -> list:
        """Balanced molecule evidence for the decider, including negative examples."""
        records, sections = self._balanced_evidence()
        out = ["Molecule evidence sample (surrogate estimates; strongest, mid-pack, and weak "
               "examples with placement spread and relaxed geometry):"]
        for title, rows in sections:
            out.append(title + ":")
            for r in rows:
                out.append(f"  {r.smiles} [{r.anchor}]  {fmt_binding_energy(r)}{self._spread(r)}"
                           + (f"  geom: {r.geometry}" if r.geometry else ""))
        if records:
            out += self._anchor_evidence_lines(records)
        return out

    def _ledger_lines(self) -> list:
        """Design-principle ledger panel."""
        principles = self.bb.ledger.all()
        lines = ["Design-principle ledger (id | status | effect | effect@matched-anchor(pool) | "
                 "n match/other | claim):"]
        if principles:
            for p in principles:
                eff = "n/a" if p.effect_size is None else f"{p.effect_size:+.1f}"
                wa = "n/a" if p.within_anchor_effect is None else f"{p.within_anchor_effect:+.1f}"
                lines.append(f"  {p.principle_id} | {p.status} | {eff} | {wa} | "
                             f"n={p.n_match}/{p.n_other} | {p.statement}")
                lines.append(f"      -> {p.detail}")
                if p.audit_notes:
                    lines.append(f"      -> audit: {p.audit_notes}")
        else:
            lines.append("  (none yet)")
        if any(p.audit_notes for p in principles):
            lines.append(
                "  Read each row's detail and audit note for its population, sample size, status, "
                "p-value, and measured confounds. Apply the system prompt's statistical definitions "
                "consistently across rows.")
            lines.append(
                "  - Anchor strata are binding-head families (the acidic or donor group that "
                "seats at the X-site), not the anchor names the proposer coins, so molecules with "
                "different anchor names but the same head are compared against each other.\n"
                "  - `n/a` in `effect@matched-anchor(pool)` means only that the contrast could not "
                "be computed for that row; the row's own audit line says why. One reason it may "
                "give is that the predicate matched almost all or almost none of the pool, so the "
                "anchor grouping was never attempted at all.")
            lines.append(
                "  - A row with no audit line was never audited. Read it as unchecked, not as "
                "clean.")
            lines.append(
                "  - On a row whose verdict came from `EXPERIMENT{goal=explain}`, `effect` was "
                "measured only on that round's freshly designed satisfy/violate pair (the n= on "
                "its `-> prospective:` line) -- a different and much smaller population than the "
                "always pool-wide `effect@matched-anchor(pool)`. Do not read those two columns "
                "as one shrinking into the other; judge that row by its own status and p-value.")
        return lines

    def _recent_action_lines(self) -> list:
        """Recent actions panel: what the decider predicted, next to what the harness
        measured.

        Without this the decider has no feedback loop on its own steps at all: `decide()`
        is a fresh run_sync with no message history, and the observation is a snapshot of
        CURRENT state, so an action that failed reads exactly like one that was never
        taken. The corrective strings the dispatcher returns ("already has a prospective
        verdict", "prospective insufficient") were being written to the decision log and
        shown to nobody, which is how a tested claim gets re-derived and how an explain
        round gets re-run against an unchanged ledger row.

        Bounded and regenerated each turn -- never a transcript -- so there is exactly one
        copy of state in the prompt and it is the current one.
        """
        entries = self.bb.log.entries[-6:]
        if not entries:
            return []
        out = ["\nRecent actions (prediction -> result). A skipped action changed no campaign "
               "state. An insufficient explain result may add molecules without updating the "
               "principle; check arm counts before requesting a top-up:"]
        for e in entries:
            label = e["action"]
            if label == "EXPERIMENT":
                label += f"[{e['goal']}/{e['control']}]"
            out.append(f"  iter {e['iter']} {label}")
            exp = _clip(e["expected_outcome"], 110)
            if exp:
                out.append(f"      you expected: {exp}")
            out.append(f"      measured: {_clip(e['result'], 190)}")
        return out

    def _nudge_line(self):
        """Dynamic balance reminder (explore/exploit/explain), or the opening-phase quota."""
        bb = self.bb
        recent = [(e["action"], e["goal"]) for e in bb.log.entries[-6:]]
        n_hyp = sum(a == "DERIVE_PRINCIPLE" or g == "explain" for a, g in recent)
        n_der = sum(a == "DERIVE_PRINCIPLE" for a, _ in recent)
        n_search = sum(a == "EXPERIMENT" and g in ("explore", "exploit") for a, g in recent)
        n_insuff = sum(p.status == "insufficient" for p in bb.ledger.all())
        # 15 molecules: what the pool needs before the balance reminder means anything -- and,
        # below it, before a derived principle can populate both groups.
        if len(bb.pool) < 15:
            # The balance reminder below is silent this early, so this is the ONLY cadence
            # guidance the opening phase gets -- and the opening phase is where an
            # under-sampled anchor space quietly caps everything the campaign can find later.
            return (
                f"Opening phase ({len(bb.pool)}/15 molecules): explore toward "
                f"~15-20 molecules spanning multiple anionic and neutral anchors, including new "
                f"anchors where useful. Retrospective validation needs "
                f">={self.retrospective_min_per_group} "
                f"molecules on each predicate side. Build this breadth before deriving a "
                f"principle; exploit only an existing hit while the anchor space remains sparse.")
        nudge = (f"Campaign balance: "
                 f"{len(bb.ledger.all())} principle(s), {len(bb.ledger.candidates())} candidate "
                 f"(in-sample) / {len(bb.ledger.supported())} prospectively supported; "
                 f"explain/derive-steps={n_hyp}, explore+exploit={n_search}.")
        if n_hyp == 0:
            nudge += (" No recent hypothesis step is recorded; consider `DERIVE_PRINCIPLE` or "
                      "`EXPERIMENT{goal=explain}`.")
        elif n_insuff >= 1 and n_der >= 2:
            nudge += (f" {n_insuff} principle(s) remain insufficient because an arm has "
                      f"<{self.retrospective_min_per_group} molecules or too few independent "
                      f"scaffold series; "
                      "the ledger identifies which, and you keep re-deriving them. Stop "
                      "re-deriving the same claim: another derivation cannot repair an arm that "
                      "is too small. Add molecules to the smaller predicate arm "
                      f"through `EXPERIMENT{{goal=explore}}` until it reaches "
                      f">={self.retrospective_min_per_group}, or derive a different, populated "
                      f"contrast. "
                      f"Scaffold coverage is not a count of SMARTS predicate arms.")
        else:
            nudge += (" Consider `EXPERIMENT{goal=exploit, control=matched}`: make single-edit "
                      "children of the strongest 1-3 binders to seek more-negative energies and "
                      "attribute changes. Continue while gains appear; stop after several exploit "
                      "rounds without improvement.")
        return nudge

    def _rejected_summary(self) -> str:
        def _tally(pairs) -> str:
            # key by reason prefix (before ':'), not the full SMARTS detail.
            c = Counter(r.split(":")[0] for _, r in pairs)
            return ", ".join(f"{k} x{v}" for k, v in c.most_common(4))

        failed = self.experiment.failed
        rejects = self.experiment.rejects
        dups = self.experiment.duplicates
        if not failed and not rejects and not dups:
            return ""
        parts = []
        if rejects:
            parts.append(f"{len(rejects)} refused before measurement ({_tally(rejects)})")
        if failed:
            parts.append(f"{len(failed)} reached the GPU but measured nothing "
                         f"({_tally(failed)}) -- these still cost budget")
        if dups:
            parts.append(f"{len(dups)} repeated evaluated molecules (not measured or charged, but "
                         f"they occupied proposal slots)")
        return ("Attrition: " + "; ".join(parts) +
                ". The proposer receives exact molecule-level reasons; allow for further losses "
                "when sizing the next round.")

    def _ingest(self, recs, charged: int) -> None:
        """Add measured records to the pool and charge the budget. ``charged`` counts every
        molecule that reached the GPU, so failures are billed too."""
        for r in recs:
            self.bb.pool.add(r)
        self.bb.budget.evals_spent += charged
        self._last_batch_recs = recs

    def _principle_context(self, exclude_id: Optional[str] = None) -> str:
        """The principle block injected into the proposer prompt and the Summarizer's input:
        conditionable() = supported + candidate."""
        sup = [p for p in self.bb.ledger.conditionable() if p.principle_id != exclude_id]
        return format_supported_principles(sup, min_anchor_stratum=self.min_anchor_stratum)

    def _evidence_block(self) -> str:
        """Balanced surrogate evidence for the proposer.

        Showing only winners makes every apparent motif look favorable and gives the proposer no
        counterexamples. Keep the context bounded while including strong, central, and weak
        outcomes, then summarize every represented anchor (up to a separate hard cap).
        """
        records, sections = self._balanced_evidence()
        if not records:
            return ""
        lines = [
            f"Measured evidence: campaign-local MatterSim+D3 surrogate relaxations, not "
            f"experiment or DFT. Differences with |ΔE_bind| < {self.effect_size_threshold:g} eV "
            f"are unresolved by policy; this threshold is not a per-value error bar. A displayed "
            f"`+-sd` is the actual spread over independent placements of that molecule:",
        ]

        for title, rows in sections:
            lines.append(title + ":")
            for r in rows:
                lines.append(f"  {r.smiles} [{r.anchor}]  {fmt_binding_energy(r)}{self._spread(r)}"
                             + (f"  geom: {r.geometry}" if r.geometry else ""))
        lines += self._anchor_evidence_lines(records)
        lines.append("Scaffold-family coverage (family: count, strongest BE):")
        for name, (c, best) in list(self.bb.pool.coverage().items())[:8]:
            lines.append(f"  {name}: n={c}, best={best:+.2f}")
        return "\n".join(lines)

    def _resolve_experiment(self, d: Decision):
        """Resolve one EXPERIMENT decision into concrete (goal, control, brief). Downgrades a
        meaningless (explore, matched) and names the strongest binders as PARENTS for exploit
        rounds so children can be single-edit. Decision's validator has already filled in the
        goal's default control."""
        goal = d.goal
        control = d.control
        if goal == "explore" and control == "matched":
            control = "free"
        brief = d.brief
        if goal == "exploit":
            best = self.bb.pool.best(3)
            if best:
                names = "; ".join(f"{r.smiles} [{r.anchor}] ({fmt_binding_energy(r)})"
                                  for r in best)
                brief = (f"Refine these strongest binders as parents; edit rather than copy them: "
                         f"{names}. {d.brief}").strip()
        return goal, control, brief

    def _current_playbook(self) -> str:
        """The learned edit playbook: the Summarizer's distillation of the accumulated directional
        hypotheses + their MEASURED outcomes, fed to both the proposer and the decider."""
        be_by = self._be_by_smiles()
        outcomes = [(r, self._direction_outcome(r, be_by)) for r in self.bb.pool.all()
                    if r.hypothesis]
        n_flat = sum(1 for _, o in outcomes if o == "flat")   # below threshold, excluded
        decided = [(r, o) for r, o in outcomes if o in ("correct", "WRONG")]
        if not decided:
            return ""                                   # nothing decided yet -> nothing to learn from
        n_correct = sum(1 for _, o in decided if o == "correct")
        n_dec = len(decided)
        balanced, majority, maj_cls = self._direction_skill(decided)
        # ground-truth tally computed here, never trusted from the LLM.
        flat_note = (f" ({n_flat} below-threshold comparison{'s' if n_flat != 1 else ''} excluded as "
                     f"unresolvable)" if n_flat else "")
        tally_line = (f"Measured tally (authoritative): {n_correct}/{n_dec} decided directional "
                      f"predictions were correct{self._skill_suffix(balanced, majority, maj_cls)}."
                      f"{flat_note} Per-family counts must be subsets of these {n_dec} instances.")
        history = self._format_history(decided, be_by)
        try:
            new_summary = self.summarizer.summarize(history, self._principle_context(), tally_line)
        except Exception as e:  # noqa: BLE001
            log.warning("summarizer failed: %s; keeping the previous playbook", e)
            return self._render_playbook(n_correct, n_dec, balanced, majority, maj_cls)
        self._last_summary = new_summary
        self._summary_n_decided = n_dec
        text = self._render_playbook(n_correct, n_dec, balanced, majority, maj_cls)
        self._append_playbook_md(n_dec, text)
        return text

    def _render_playbook(self, n_correct: int, n_dec: int,
                         balanced, majority, maj_cls) -> str:
        if self._last_summary is None:
            return ""
        overall = (f"Measured directional accuracy (computed): "
                   f"{n_correct}/{n_dec} decided predictions correct"
                   f"{self._skill_suffix(balanced, majority, maj_cls)}.")
        if self._summary_n_decided != n_dec:
            overall += (f" (This line is current. The playbook below was distilled from the "
                        f"first {self._summary_n_decided} of these {n_dec} decided proposals, so "
                        f"its per-family counts are older; trust this line where they disagree.)")
        return format_playbook(self._last_summary, overall_line=overall,
                               effect_size_threshold=self.effect_size_threshold)

    def _append_playbook_md(self, n_decided: int, text: str) -> None:
        """Append the freshly-refreshed learned playbook to playbook_log.md for auditability."""
        path = os.path.join(self.log_dir, "playbook_log.md")
        first = not os.path.exists(path)
        with open(path, "a") as fh:
            if first:
                fh.write("# Learned edit playbook (Summarizer)\n")
            fh.write(f"\n## Iter {self.bb.budget.iters_spent} (from {n_decided} decided proposals)\n\n"
                     f"{text}\n")

    def _format_history(self, decided, be_by) -> str:
        """One line per decided proposal for the summarizer: outcome, anchor, measured BE,
        Δ vs reference, predicted direction, and the full hypothesis (the edit tested)."""
        lines = []
        for r, outcome in decided:
            dv = r.binding_energy - self._ref_be(be_by, r.parent_smiles)
            lines.append(
                f"[{outcome}] anchor={r.anchor} measured={_fmt_opt(r.binding_energy)}eV "
                f"Δvs_ref={dv:+.2f} pred={r.predicted_direction} :: {r.hypothesis}")
        return "\n".join(lines)

    # actions
    def _dispatch(self, d: Decision, playbook: str) -> str:
        """Route one decision to its handler. STOP is control flow (ends the loop)."""
        self.experiment.iteration = self.bb.budget.iters_spent
        if d.action == "STOP":
            return "stop"
        if d.action == "EXPERIMENT":
            return self._run_experiment(d, playbook)
        return self._run_derive_principle(d)    # DERIVE_PRINCIPLE, the only action left

    def _run_experiment(self, d: Decision, playbook: str) -> str:
        goal, control, brief = self._resolve_experiment(d)
        charged_before = self.experiment.evals_charged
        if goal == "explain":
            p = self.bb.ledger.get(d.principle_id)
            if p is None:
                return f"explain: no principle {d.principle_id!r} in ledger; skipped"
            recs = self.experiment.propose_for_predicate(
                p, max(2, d.n), matched=(control == "matched"),
                principle_context=self._principle_context(exclude_id=p.principle_id),
                evidence=self._evidence_block(), playbook=playbook)
            self._ingest(recs, charged=self.experiment.evals_charged - charged_before)
            verdict = prospective_verdict(p, recs,
                                          min_per_group=self.prospective_min_per_group,
                                          effect_size_threshold=self.effect_size_threshold,
                                          significance_alpha=self.significance_alpha)
            verdict = self._measure_confounds(verdict)
            if verdict.status == "insufficient":
                return (f"explain[{control}] {p.principle_id} -> prospective insufficient "
                        f"({verdict.detail}); prior {p.status} retained; "
                        f"best now {self.bb.pool.best_value()}")
            self.bb.ledger.add(verdict)
            return (f"explain[{control}] {p.principle_id} -> {verdict.status} "
                    f"({verdict.detail}); best now {self.bb.pool.best_value()}")
        # explore / exploit
        recs = self.experiment.propose_families(brief, max(1, d.n), goal=goal, control=control,
                                                principle_context=self._principle_context(),
                                                evidence=self._evidence_block(), playbook=playbook)
        charged = self.experiment.evals_charged - charged_before
        self._ingest(recs, charged=charged)
        lost = charged - len(recs)
        return (f"{goal}[{control}]→evaluated {len(recs)}"
                + (f" ({lost} measured nothing)" if lost > 0 else "")
                + f"; best now {self.bb.pool.best_value()}")

    def _run_derive_principle(self, d: Decision) -> str:
        bb = self.bb
        if d.principle is None:
            return "no principle supplied; skipped"
        d.principle = d.principle.model_copy(update={
            f: DesignPrinciple.model_fields[f].get_default() for f in _VALIDATOR_OWNED})
        prior = bb.ledger.get(d.principle.principle_id)
        if prior is not None and prior.validation == "prospective":
            return (f"principle {prior.principle_id} already has a prospective verdict "
                    f"({prior.status}); a re-derive would replace it with an in-sample one. "
                    "Derive a different claim, or use EXPERIMENT{goal=explain} to re-test it.")
        validated = retrospective_verdict(d.principle, bb.pool.all(),
                                          min_per_group=self.retrospective_min_per_group,
                                          rng_seed=self.random_seed,
                                          effect_size_threshold=self.effect_size_threshold,
                                          significance_alpha=self.significance_alpha)
        # keep the in-sample verdict readable after a prospective one later overwrites it
        validated.retrospective_effect_size = validated.effect_size
        validated.retrospective_p_value = validated.p_value
        validated.retrospective_status = validated.status
        validated = self._measure_confounds(validated)
        bb.ledger.add(validated)
        return f"principle {validated.principle_id} -> {validated.status} ({validated.detail})"

    def _measure_confounds(self, p):
        """Attach the confound MEASUREMENTS. Called only on a freshly-returned verdict, so
        `insufficient` is the one status that still has nothing to audit."""
        if p.status == "insufficient":
            return p
        return audit_principle(p, self.bb.pool.all(), min_stratum=self.min_anchor_stratum)

    def _agent_usages(self) -> dict:
        """Cumulative token usage per LLM agent (decider / proposer / summarizer)."""
        return {name: dict(agent.usage) for name, agent in (
            ("decider", self.decider), ("proposer", self.experiment.proposer),
            ("summarizer", self.summarizer))}

    def _log_tokens(self, it: int, before: dict) -> None:
        """Per-iteration input/output token usage across the three agents."""
        after = self._agent_usages()
        per, ti, to = {}, 0, 0
        for k in after:
            di = after[k]["input"] - before[k]["input"]
            do = after[k]["output"] - before[k]["output"]
            per[k] = {"input": di, "output": do, "calls": after[k]["calls"] - before[k]["calls"]}
            ti += di; to += do
        cum_in = sum(after[k]["input"] for k in after)
        cum_out = sum(after[k]["output"] for k in after)
        log.info("[tokens] iter %d: in=%d out=%d (decider %d/%d, proposer %d/%d, summarizer %d/%d) "
                 "| cumulative in=%d out=%d", it, ti, to,
                 per["decider"]["input"], per["decider"]["output"],
                 per["proposer"]["input"], per["proposer"]["output"],
                 per["summarizer"]["input"], per["summarizer"]["output"], cum_in, cum_out)
        with open(os.path.join(self.log_dir, "token_log.jsonl"), "a") as fh:
            fh.write(json.dumps({"iter": it, "t": time.time(),
                                 "iter_input": ti, "iter_output": to,
                                 "by_agent": per, "cumulative_input": cum_in,
                                 "cumulative_output": cum_out}) + "\n")

    def run(self) -> None:
        bb = self.bb
        while bb.budget.can_continue():
            tok_before = self._agent_usages()
            playbook = self._current_playbook()
            obs = self._observe(playbook)
            self._save_observation(bb.budget.iters_spent, obs)
            try:
                decision = self.decider.decide(obs)
            except Exception as e:  # noqa: BLE001
                log.warning("decider failed: %s; stopping.", e)
                self._stop_reason = f"decider failed: {type(e).__name__}: {e}"
                break
            if decision is None:
                # A schema failure is one lost iteration, not a lost campaign: iterations
                # have never been the binding budget, evaluations have. Only a decider that
                # cannot produce ANY valid step is worth stopping for.
                self._no_decision += 1
                log.warning("decider produced no valid Decision (%d consecutive); skipping "
                            "iteration %d", self._no_decision, bb.budget.iters_spent)
                bb.budget.iters_spent += 1
                if self._no_decision >= 3:   # consecutive failures before the run is abandoned
                    self._stop_reason = (f"decider produced no valid Decision "
                                         f"{self._no_decision} iterations in a row")
                    break
                continue
            self._no_decision = 0
            self._last_batch_recs = None
            try:
                summary = self._dispatch(decision, playbook)
            except Exception as e:  # noqa: BLE001
                summary = f"action {decision.action} errored: {type(e).__name__}: {e}"
                log.warning(summary)
            # "batch" is None when this step ran no experiment (e.g. DERIVE_PRINCIPLE)
            convergence_stats = {"batch": (binding_energy_stats(self._last_batch_recs)
                                           if self._last_batch_recs is not None else None),
                                 "pool": binding_energy_stats(bb.pool.all())}
            bb.log.append(bb.budget.iters_spent, decision, summary,
                          convergence_stats=convergence_stats)
            try:
                self._log_reasoning(decision, summary, convergence_stats)
                self._log_tokens(bb.budget.iters_spent, tok_before)
            except Exception as e:  # noqa: BLE001
                log.warning("per-iteration logging failed (continuing): %s: %s",
                            type(e).__name__, e)
            finally:
                bb.budget.iters_spent += 1
            self._finalize()
            if decision.action == "STOP":
                self._stop_reason = "agent emitted STOP"
                break
            if self._plateaued(decision):
                break
        else:
            self._stop_reason = "budget exhausted (eval/iter cap)"
        self._finalize()

    def _plateaued(self, decision) -> bool:
        """Track progress and report whether the run should STOP."""
        if self.patience <= 0 or decision.action != "EXPERIMENT":
            return False
        best = self.bb.pool.best(1)
        if not best:
            return False   # no molecules evaluated yet -- nothing to improve on
        cur = rank_key(best[0])
        if self._best_loss is None or cur < self._best_loss - self.effect_size_threshold:
            self._best_loss = cur
            self._no_improve = 0
            return False
        self._no_improve += 1
        if self._no_improve >= self.patience:
            self._stop_reason = (f"early stop: best did not improve by "
                                 f">{self.effect_size_threshold} eV for {self._no_improve} "
                                 f"iterations (patience={self.patience})")
            log.info("[early-stop] %s", self._stop_reason)
            return True
        return False

    def _be_by_smiles(self) -> dict:
        return {r.smiles: r.binding_energy for r in self.bb.pool.all()}

    @staticmethod
    def _ref_be(be_by, parent_smiles):
        """Binding energy of the reference molecule, or None."""
        if not parent_smiles:
            return None
        be = be_by.get(parent_smiles)
        if be is None:
            # same identity function the pool is keyed by, so a stereo-bearing or
            # non-canonical parent_smiles still resolves to its measured record
            be = be_by.get(canonical_identity(parent_smiles))
        return be

    def _direction_outcome(self, rec, be_by_smiles: dict) -> str:
        ref_be = self._ref_be(be_by_smiles, rec.parent_smiles)
        if ref_be is None:
            return "n/a"
        if abs(rec.binding_energy - ref_be) < self.effect_size_threshold:
            return "flat"       # below the effect-size threshold
        actual = "increase" if rec.binding_energy < ref_be else "decrease"
        return "correct" if actual == rec.predicted_direction else "WRONG"

    @staticmethod
    def _direction_skill(decided):
        """Chance-corrected read of the directional tally."""
        tot_by, corr_by = {}, {}
        for r, o in decided:
            pd = r.predicted_direction
            is_ok = (o == "correct")
            # what actually happened to binding
            realized = pd if is_ok else ("decrease" if pd == "increase" else "increase")
            tot_by[realized] = tot_by.get(realized, 0) + 1
            corr_by[realized] = corr_by.get(realized, 0) + (1 if is_ok else 0)
        n = sum(tot_by.values())
        majority = max(tot_by.values()) / n
        maj_cls = max(tot_by, key=tot_by.get)
        balanced = sum(corr_by[k] / tot_by[k] for k in tot_by) / len(tot_by)
        return balanced, majority, maj_cls

    @staticmethod
    def _skill_suffix(balanced, majority, maj_cls) -> str:
        return (f" · balanced {balanced:.0%} vs 50% no-skill (outcomes were {majority:.0%} "
                f"'{maj_cls}'; that prevalence sets how easy a raw tally looks, not how much "
                f"skill it shows)")

    def _log_reasoning(self, decision, summary, convergence_stats) -> None:
        """Human-readable trace of what the LLM was thinking this step."""
        recs = self._last_batch_recs or []
        be_by = self._be_by_smiles()
        log.info("[iter %d] %s -> %s", self.bb.budget.iters_spent, decision.action, summary)
        if decision.hypothesis or decision.rationale:
            log.info("    thinking: %s | %s", _clip(decision.hypothesis, 160),
                     _clip(decision.rationale, 160))
        for r in recs:
            log.info("    %s (%s)  pred=%s vs %s / measured %s [%s]  | %s", r.smiles, r.anchor,
                     r.predicted_direction, _clip(r.parent_smiles or "field", 20),
                     _fmt_opt(r.binding_energy), self._direction_outcome(r, be_by),
                     _clip(r.rationale or r.hypothesis, 120))
        self._append_reasoning_md(decision, summary, convergence_stats, recs, be_by)

    def _append_reasoning_md(self, decision, summary, convergence_stats, recs, be_by) -> None:
        path = os.path.join(self.log_dir, "reasoning_log.md")
        first = not os.path.exists(path)
        with open(path, "a") as fh:
            if first:
                fh.write("# Discovery reasoning log\n")
            label = decision.action
            if decision.action == "EXPERIMENT":
                label += f"[{decision.goal}/{decision.control}]"
            fh.write(f"\n---\n\n## Iter {self.bb.budget.iters_spent} — {label}\n\n")
            fh.write(f"- **Hypothesis:** {decision.hypothesis}\n")
            fh.write(f"- **Rationale:** {decision.rationale}\n")
            fh.write(f"- **Expected:** {decision.expected_outcome}\n")
            fh.write(f"- **Result:** {summary}\n")
            b = convergence_stats["batch"]
            if b and b["n"]:
                fh.write(f"- **Batch:** n={b['n']}, strongest={_fmt_opt(b['best'], 3)} eV, "
                         f"mean={_fmt_opt(b['mean'], 3)} eV\n")
            if recs:
                # One block per molecule so the FULL hypothesis is never clipped.
                fh.write("\n### Molecules (full hypotheses)\n\n")
                for i, r in enumerate(recs, 1):
                    fh.write(f"**{i}. `{r.smiles}`**  ·  {r.anchor}  ·  "
                             f"pred **{r.predicted_direction}** vs `{r.parent_smiles or '—'}`"
                             f"  ·  measured **{_fmt_opt(r.binding_energy, 3)} eV**  ·  "
                             f"dir? **{self._direction_outcome(r, be_by)}**\n")
                    if r.geometry:
                        fh.write(f"  · geom: `{r.geometry}`\n")
                    fh.write(f"\n  > {r.hypothesis or r.rationale or '(no hypothesis)'}\n\n")

    def summary(self) -> dict:
        bb = self.bb
        supported, candidates = bb.ledger.supported(), bb.ledger.candidates()

        def rows(principles):
            return [{"id": p.principle_id, "statement": p.statement,
                     "effect_size": p.effect_size, "validation": p.validation,
                     "n_match": p.n_match, "n_other": p.n_other} for p in principles]

        return {
            "objective": OBJECTIVE_LABEL,
            "best_metric": bb.pool.best_value(),
            "n_evaluated": len(bb.pool),
            "n_families": len(bb.pool.coverage()),
            "n_principles": len(bb.ledger.all()),
            "n_supported": len(supported),
            "supported_principles": rows(supported),
            "n_candidate": len(candidates),
            "candidate_principles": rows(candidates),
            "best_molecules": [
                {"smiles": r.smiles, "anchor": r.anchor,
                 "binding_energy": r.binding_energy, "metric": r.binding_energy}
                for r in bb.pool.best(10)],
            "decisions": len(bb.log.entries),
            "stop_reason": self._stop_reason,
            "stopped_early": bool(self._stop_reason and self._stop_reason.startswith("early stop")),
            "patience": self.patience,
            "effect_size_threshold": self.effect_size_threshold,
        }

    def _atomic_write(self, name: str, emit) -> None:
        path = os.path.join(self.log_dir, name)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as fh:
                emit(fh)
                fh.flush()
                os.fsync(fh.fileno())  # force to disk before the rename (Lustre-safe)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)         # a failed write leaves no debris; `path` is never touched
            except OSError:
                pass
            raise

    def _finalize(self) -> None:
        def _decisions(fh):
            for e in self.bb.log.entries:
                fh.write(json.dumps(e) + "\n")

        def _convergence(fh):
            for e in self.bb.log.entries:
                fh.write(json.dumps({"iter": e["iter"], "action": e["action"],
                                     "convergence_stats": e["convergence_stats"]}) + "\n")

        def _principles(fh):
            for p in self.bb.ledger.all():
                fh.write(p.model_dump_json() + "\n")

        def _candidates(fh):
            be_by = self._be_by_smiles()
            for r in self.bb.pool.all():
                fh.write(json.dumps({"smiles": r.smiles, "anchor": r.anchor,
                                     # anchor the proposer DECLARED, before any same-mode
                                     # fallback relabelled it -- without this the relabel
                                     # rate is unmeasurable after the fact.
                                     "declared_anchor": r.declared_anchor,
                                     "binding_energy": r.binding_energy,
                                     "binding_energy_std": r.binding_energy_std,
                                     "n_placements_ok": r.n_placements_ok,
                                     "n_placements_reacted": r.n_placements_reacted,
                                     "geometry": r.geometry,
                                     "hypothesis": r.hypothesis,
                                     "predicted_direction": r.predicted_direction,
                                     "parent_smiles": r.parent_smiles,
                                     "direction_correct": self._direction_outcome(r, be_by),
                                     "rationale": r.rationale,
                                     "murcko": r.murcko, "source": r.source}) + "\n")

        def _rejected_jsonl(fh):
            for r in self.experiment.rejected_log:
                fh.write(json.dumps(r) + "\n")

        def _rejected_md(fh):
            """Human-readable companion to reasoning_log.md."""
            recs = self.experiment.rejected_log
            fh.write("# Rejected / lost molecules\n")
            if not recs:
                fh.write("\n_Nothing lost yet._\n")
                return
            cur = object()
            for r in recs:
                if r["iter"] != cur:
                    cur = r["iter"]
                    fh.write(f"\n---\n\n## Iter {cur}\n\n")
                fh.write(f"**`{r['smiles']}`** · **{r['kind']}** · `{r['reason']}`"
                         + (f" · {r['anchor']}" if r["anchor"] else "")
                         + f" · {r['source']}\n")
                if r["parent_smiles"]:
                    fh.write(f"  · edit of `{r['parent_smiles']}`, "
                             f"predicted **{r['predicted_direction']}**\n")
                fh.write(f"\n  > {r['hypothesis'] or r['rationale'] or '(no hypothesis)'}\n\n")

        def _anchors(fh):
            for spec in self.experiment.registry.specs():
                fh.write(spec.model_dump_json() + "\n")

        for name, emit in (
            ("decision_log.jsonl", _decisions),
            ("convergence.jsonl", _convergence),
            ("principles.jsonl", _principles),
            ("candidates.jsonl", _candidates),
            ("rejected_log.jsonl", _rejected_jsonl),
            ("rejected_log.md", _rejected_md),
            ("anchors.jsonl", _anchors),
            ("summary.json", lambda fh: json.dump(self.summary(), fh, indent=2)),
        ):
            try:
                self._atomic_write(name, emit)
            except Exception as e:  # noqa: BLE001
                log.warning("could not persist %s (continuing; will retry next iteration): "
                            "%s: %s", name, type(e).__name__, e)
