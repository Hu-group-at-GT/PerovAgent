from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from perovagent.agents.structured_agent import StructuredAgent


class EditLesson(BaseModel):
    """One aggregated structural edit lesson learned from the proposal history."""
    move: str = Field(description=(
        "Directional structural edit, written consistently as feature A -> feature B."))
    n_correct: int = 0
    n_wrong: int = 0
    verdict: Literal["reliable_help", "reliable_hurt", "noisy_or_neutral"] = Field(description=(
        "Measured outcome for the stated A -> B structural edit: reliable_help strengthened "
        "binding; reliable_hurt weakened it; noisy_or_neutral was unresolved or inconsistent."))
    takeaway: str


class Summary(BaseModel):
    """Distilled learning fed to the proposer."""
    edit_lessons: list[EditLesson] = Field(default_factory=list)
    calibration_note: str = ""
    do_next: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)


def format_playbook(summary: Summary, overall_line: str, *, effect_size_threshold: float) -> str:
    """Format a Summary into the text block injected into the proposer/decider prompt.
    ``effect_size_threshold`` (eV) is quoted to the agents, so it must be the configured value."""
    lines = ["Learned edit playbook (from prior proposals and measured outcomes; "
             f"|ΔE_bind| < {effect_size_threshold:g} eV is treated as unresolved, so only "
             "aggregated directional edit families are shown):",
             "  " + overall_line]
    order = {"reliable_help": 0, "reliable_hurt": 1, "noisy_or_neutral": 2}
    label = {"reliable_help": "A->B helps", "reliable_hurt": "A->B hurts",
             "noisy_or_neutral": "A->B noisy/neutral"}
    for les in sorted(summary.edit_lessons, key=lambda x: order[x.verdict]):
        n = les.n_correct + les.n_wrong
        cnt = f" ({les.n_correct}/{n} correct)" if n > 0 else ""
        lines.append(f"  [{label[les.verdict]}] {les.move}{cnt} — {les.takeaway}")
    if summary.calibration_note:
        lines.append(f"Prediction calibration: {summary.calibration_note}")
    if summary.do_next:
        lines.append("Try next: " + "; ".join(summary.do_next))
    if summary.avoid:
        lines.append("Avoid: " + "; ".join(summary.avoid))
    return "\n".join(lines)


class LLMSummarizer(StructuredAgent):
    _output_type = Summary
    _preflight_probe = "Health check: return an empty Summary."

    def __init__(self, model: str, effort: str | None, *, effect_size_threshold: float):
        super().__init__(model, effort, summarizer_system_prompt(effect_size_threshold))

    def summarize(self, history: str, principles: str, tally_line: str) -> Summary:
        parts = []
        if principles:
            parts.append("Conditionable design principles (`candidate` is in-sample; `supported` "
                         "is prospectively confirmed):\n" + principles)
        parts.append(tally_line)
        parts.append("Proposal history (edit hypothesis -> measured directional outcome):\n" + history)
        parts.append("Distill the learned edit playbook now.")
        result = self.agent.run_sync("\n\n".join(parts))
        self._accumulate(result)
        return result.output


def summarizer_system_prompt(effect_size_threshold: float) -> str:
    return f"""\
You summarize learning in a molecular-discovery loop. A proposer supplies ligand hypotheses and
predicts whether each edit will increase or decrease binding strength relative to a reference.
MatterSim+D3 surrogate estimates then classify each prediction as correct, wrong, or unresolved.

Use the accumulated hypothesis, direction, anchor, ΔE_bind, outcome history, and conditionable
principles to produce a compact edit playbook. Summarize measured patterns; do not invent molecules,
predict energies, or derive new principles.

Rules
- Group free-text hypotheses into a few edit families, such as distal or ortho/peri substitution,
  anchor swaps, chain extension, ring fusion, or rigidification. Report one lesson per family.
- Write each `move` consistently as A -> B. `reliable_help` means the measured edit generally made
  E_bind more negative; `reliable_hurt` means it made E_bind less negative. These labels describe
  measured edit effects. Use `n_correct` and `n_wrong` only for prediction accuracy.
- Comparisons with |ΔE_bind| < {effect_size_threshold:g} eV are already excluded as unresolved. Require
  several resolved instances (aim for at least 3) with consistent effects before using
  `reliable_help` or `reliable_hurt`; otherwise use `noisy_or_neutral`. Do not infer a rule from
  1-2 examples.
- Note confounding when an apparent edit effect co-varies with anchor identity. If the strong
  instances all share one anchor, say so in that family's `takeaway` rather than crediting the
  edit.
- In `calibration_note`, assess the reliability of the proposer's directional predictions and
  identify where measured geometry should outweigh qualitative intuition. This calibration is
  often the single most useful lesson in the digest.
- Provide 2-5 short, evidence-based entries in `do_next` and 2-5 repeatedly unsuccessful moves in
  `avoid` when the history supports them.
- Treat the supplied measured tally as authoritative. Family counts must refer to listed instances,
  and each family's `n_correct + n_wrong` must equal its assigned listed proposals. Counts must not
  exceed the overall correct/wrong totals or collectively exceed the number of decided predictions.
  Use zero when a count is uncertain. Do not restate the overall accuracy fraction; it is computed
  separately.

Be terse and honest. If evidence is sparse, return few or no lessons and state that limitation in
`calibration_note`."""
