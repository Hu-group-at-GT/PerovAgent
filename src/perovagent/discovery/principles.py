from __future__ import annotations

import hashlib
from math import comb
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, field_validator
from scipy.stats import mannwhitneyu, wilcoxon
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

from perovagent.chemistry.admission import canonical_identity

RDLogger.DisableLog("rdApp.*")


class DesignPrinciple(BaseModel):
    """A falsifiable, generalizable structure/geometry -> binding-energy claim derived
    by the agent from candidate observations, plus its empirical validation verdict."""
    statement: str
    predicate: str
    predicted_direction: Literal["lower", "higher"] = "lower"
    rationale: str = ""
    status: Literal["proposed", "candidate", "supported", "weak", "refuted", "insufficient"] = "proposed"
    effect_size: Optional[float] = None
    n_match: int = 0
    n_other: int = 0
    p_value: Optional[float] = None
    n_match_eff: int = 0
    n_other_eff: int = 0
    n_pairs: int = 0
    effect_sd: Optional[float] = None
    validation: Literal["none", "retrospective", "prospective"] = "none"
    detail: str = ""
    retrospective_effect_size: Optional[float] = None
    retrospective_p_value: Optional[float] = None
    retrospective_status: str = ""
    within_anchor_effect: Optional[float] = None
    n_usable_anchor_strata: int = 0
    within_size_effect: Optional[float] = None
    frac_match: float = 0.0
    degenerate_partition: bool = False
    anchor_perfectly_collinear: bool = False
    audit_notes: str = ""

    @field_validator("predicate")
    @classmethod
    def _predicate_parses_as_smarts(cls, v: str) -> str:
        """Reject an unparseable predicate at emission and report likely syntax causes."""
        if not v.strip():
            raise ValueError("predicate must be a non-empty SMARTS pattern")
        if Chem.MolFromSmarts(v) is None:
            raise ValueError(
                f"predicate {v!r} is not a parseable SMARTS pattern. Check that ring-bond digits "
                "are paired (a lone '1' in 'c1cc' is unclosed), brackets and parentheses balance, "
                "and the value is bare SMARTS rather than SMILES or quoted text. Return a "
                "corrected pattern; do not resend this one.")
        return v

    @property
    def principle_id(self) -> str:
        return hashlib.sha1(self.statement.encode("utf-8")).hexdigest()[:8]


def satisfies(predicate: str, smiles: str) -> Optional[bool]:
    """True/False if the molecule satisfies the predicate SMARTS, None if unparseable
    or the predicate is malformed."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    patt = Chem.MolFromSmarts(predicate) if predicate else None
    if patt is None:
        return None
    return mol.HasSubstructMatch(patt)


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:+.2f}"


def predicate_text(p: str) -> str:
    """Human/prompt-readable form of a predicate."""
    return f"matches SMARTS '{p}'"


def format_supported_principles(principles, *, min_anchor_stratum: int) -> str:
    if not principles:
        return ""
    lines = ["Conditionable design principles -- condition proposals on these, and weigh "
             "each by its status: a `candidate` principle is in-sample and not yet "
             "prospectively confirmed:"]
    # Prospective evidence outranks an impressive in-sample effect. Within each evidence tier,
    # retain the strongest effects so the block stays compact.
    top = sorted(
        principles,
        key=lambda p: (0 if p.status == "supported" else 1, -p.effect_size),
    )[:5]
    for p in top:
        eff = _fmt(p.effect_size)
        pred = predicate_text(p.predicate)
        # Label both tiers explicitly so the status never depends on an absent marker.
        marks = [f"status={p.status}"]
        if p.within_anchor_effect is not None:
            marks.append(f"matched-anchor effect={_fmt(p.within_anchor_effect)}")
        elif p.degenerate_partition:
            marks.append("matched-anchor effect not estimated: predicate matches ~all or "
                         "~none of the pool, so no anchor-grouped contrast was measured")
        elif p.anchor_perfectly_collinear:
            marks.append("matched-anchor effect unavailable: predicate is collinear with the "
                         "binding-head family, supporting an anchor-level choice but not "
                         "differences within that head")
        else:
            marks.append(f"matched-anchor effect not yet estimable: a mixed family exists, "
                         f"but none has >={min_anchor_stratum} molecules per side; "
                         f"underpowered and not evidence for an anchor-level effect")
        flag = "; " + "; ".join(marks)
        direction = ("matching molecules bind more strongly (more negative E_bind)"
                     if p.predicted_direction == "lower" else
                     "matching molecules bind more weakly (less negative E_bind)")
        lines.append(f"  - {p.statement} [{pred}; {direction}; effect {eff}{flag}]")
    return "\n".join(lines)


def _split_by_predicate(principle, records):
    """Split records into (match, other) by the predicate, skipping records the predicate
    cannot be evaluated on."""
    match, other = [], []
    for r in records:
        m = satisfies(principle.predicate, r.smiles)
        if m is None:
            continue
        (match if m else other).append(r)
    return match, other


def min_achievable_p(n_match: int, n_other: int) -> float:
    """Smallest one-sided Mann-Whitney p attainable at these group sizes = 1 / C(n_match+n_other,
    n_match), reached only under perfect separation."""
    return 1.0 / comb(n_match + n_other, n_match)


def _values_by_predicate(principle, records):
    """(binding energies of matching molecules, of the rest)."""
    match, other = _split_by_predicate(principle, records)
    return [r.binding_energy for r in match], [r.binding_energy for r in other]


def _signed_effect(principle, match, other) -> float:
    """median(other) - median(match) in the predicted direction."""
    eff = float(np.median(other) - np.median(match))
    return eff if principle.predicted_direction == "lower" else -eff


def _mannwhitney(principle, match, other) -> Optional[float]:
    """One-sided Mann-Whitney p-value in the predicted direction."""
    alt = "less" if principle.predicted_direction == "lower" else "greater"
    try:
        return float(mannwhitneyu(match, other, alternative=alt).pvalue)
    except ValueError:
        return None


def _matched_pairs(principle, records) -> list[tuple[float, float]]:
    """(match_value, other_value) for every resolved matched pair."""
    by = {}
    for r in records:
        key = canonical_identity(r.smiles)
        if key:
            by.setdefault(key, r)
    pairs, used = [], set()
    for r in records:
        sat = satisfies(principle.predicate, r.smiles)
        if sat is None:
            continue
        twin = by.get(canonical_identity(r.parent_smiles))
        if twin is None or satisfies(principle.predicate, twin.smiles) is not (not sat):
            continue
        m_rec, o_rec = (r, twin) if sat else (twin, r)
        # unordered key: a mutually-linked pair is one observation, not two
        key = frozenset((canonical_identity(m_rec.smiles), canonical_identity(o_rec.smiles)))
        if key in used:
            continue
        used.add(key)
        pairs.append((m_rec.binding_energy, o_rec.binding_energy))
    return pairs


def _signed_pair_deltas(principle, pairs) -> list[float]:
    """Per-pair effect in the predicted direction; positive = the predicate helped."""
    sign = 1.0 if principle.predicted_direction == "lower" else -1.0
    return [sign * (o - m) for m, o in pairs]


def _wilcoxon_paired(deltas) -> Optional[float]:
    """One-sided signed-rank p that the per-pair effect is positive."""
    try:
        return float(wilcoxon(deltas, alternative="greater").pvalue)
    except ValueError:
        return None


_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _independent_units(smiles_values):
    """Collapse near-duplicate molecules (Butina on Morgan fingerprints) into one independent
    unit each and return the unit medians."""
    fps = [_MORGAN.GetFingerprint(Chem.MolFromSmiles(s)) for s, _ in smiles_values]
    vals = [v for _, v in smiles_values]
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
    # Tanimoto >= 0.6 (Morgan r2) is one independent unit: the Butina "same series" cutoff
    return [float(np.median([vals[j] for j in cluster]))
            for cluster in Butina.ClusterData(dists, len(fps), 1.0 - 0.6, isDistData=True)]


def _unit_medians_by_predicate(principle, records):
    """Partition records by predicate, then within each group collapse near-duplicates to one
    independent unit."""
    match, other = _split_by_predicate(principle, records)
    return (_independent_units([(r.smiles, r.binding_energy) for r in match]),
            _independent_units([(r.smiles, r.binding_energy) for r in other]))


def _halves_reproduce_sign(principle, match_units, other_units, rng_seed) -> tuple[bool, bool]:
    if min(len(match_units), len(other_units)) < 4:
        return True, False
    rng = np.random.RandomState(rng_seed)
    match_shuf = [match_units[i] for i in rng.permutation(len(match_units))]
    other_shuf = [other_units[i] for i in rng.permutation(len(other_units))]
    for match_half, other_half in ((match_shuf[::2], other_shuf[::2]),
                                   (match_shuf[1::2], other_shuf[1::2])):
        if _signed_effect(principle, match_half, other_half) <= 0:
            return False, True
    return True, True


def retrospective_verdict(principle: DesignPrinciple, records, effect_size_threshold: float,
                          min_per_group: int, significance_alpha: float,
                          rng_seed: int) -> DesignPrinciple:
    """Retrospectively validate a principle against the pool it was derived from, with a
    within-sample random half split."""
    match, other = _values_by_predicate(principle, records)
    p = principle.model_copy()
    p.n_match, p.n_other = len(match), len(other)
    p.validation = "retrospective"

    if min(len(match), len(other)) < min_per_group:
        p.status = "insufficient"
        p.detail = f"need >={min_per_group}/arm, have {len(match)}/{len(other)}"
        return p

    # effect + significance on independent-unit medians, not per molecule
    match_units, other_units = _unit_medians_by_predicate(principle, records)
    p.n_match_eff, p.n_other_eff = len(match_units), len(other_units)

    # <2 independent units on a side cannot be tested (one unit is a single observation).
    min_p = min_achievable_p(len(match_units), len(other_units))
    eff = (_signed_effect(principle, match_units, other_units)
           if (match_units and other_units) else None)

    if min(len(match_units), len(other_units)) < 2 or (eff > 0 and min_p >= significance_alpha):
        p.effect_size = eff
        p.status = "insufficient"
        p.detail = (f"only {len(match_units)}/{len(other_units)} independent series "
                    f"(from {len(match)}/{len(other)} molecules) -- "
                    + ("too few distinct families to test"
                       if min(len(match_units), len(other_units)) < 2
                       else f"minimum attainable p={min_p:.4f} >= alpha={significance_alpha}; "
                            f"these arm sizes cannot meet the threshold"))
        return p

    pval = _mannwhitney(principle, match_units, other_units)
    p.effect_size, p.p_value = eff, pval

    halves_ok, tested = _halves_reproduce_sign(principle, match_units, other_units, rng_seed)

    sig = pval is not None and pval < significance_alpha
    if eff <= 0:
        p.status = "refuted"
    elif eff > effect_size_threshold and halves_ok and sig:
        p.status = "candidate"
    else:
        p.status = "weak"
    ho = "replicated" if (halves_ok and tested) else ("n/a" if not tested else "not replicated")
    p.detail = (f"cluster effect={eff:+.2f} eV (decision threshold {effect_size_threshold}), "
                f"p={pval if pval is None else round(pval, 4)} (series-level), "
                f"held-out={ho}, n={len(match)}/{len(other)} molecules, "
                f"n_eff={len(match_units)}/{len(other_units)} series")
    return p


def prospective_verdict(principle: DesignPrinciple, records, effect_size_threshold: float,
                        significance_alpha: float,
                        min_per_group: int) -> DesignPrinciple:
    """Verdict on a fresh designed-experiment set; only this tier can set `supported`."""
    match, other = _values_by_predicate(principle, records)
    p = principle.model_copy()
    p.n_match, p.n_other = len(match), len(other)
    p.validation = "prospective"

    if min(len(match), len(other)) < min_per_group:
        p.status = "insufficient"
        p.detail = f"prospective: need >={min_per_group}/arm, have {len(match)}/{len(other)}"
        return p

    match_units, other_units = _unit_medians_by_predicate(principle, records)
    p.n_match_eff, p.n_other_eff = len(match_units), len(other_units)

    deltas = _signed_pair_deltas(principle, _matched_pairs(principle, records))
    p.n_pairs = len(deltas)

    if deltas:
        eff = float(np.mean(deltas))
        p.effect_sd = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None
        min_p = 0.5 ** len(deltas)
        pval = _wilcoxon_paired(deltas)
        basis = f"paired n={len(deltas)} twin pairs"
    else:
        eff = _signed_effect(principle, match_units, other_units)
        min_p = min_achievable_p(len(match_units), len(other_units))
        pval = _mannwhitney(principle, match_units, other_units)
        basis = (f"unpaired (no twins resolved) n_eff={len(match_units)}/{len(other_units)} "
                 f"series from {len(match)}/{len(other)} molecules")

    p.effect_size, p.p_value = eff, pval
    if eff > 0 and min_p >= significance_alpha:
        p.status = "insufficient"
        p.detail = (f"prospective: {basis}; minimum attainable p={min_p:.4f} >= "
                    f"alpha={significance_alpha}, so these arm sizes cannot meet the threshold")
        return p

    sig = pval is not None and pval < significance_alpha
    if eff <= 0:
        p.status = "refuted"
    elif eff > effect_size_threshold and sig:
        p.status = "supported"
    else:
        p.status = "weak"
    spread = f" +-{p.effect_sd:.2f}" if p.effect_sd is not None else ""
    at_min_p = ("; at the resolution floor: this is the smallest p available to the design, so "
                "the pair count limits the evidence"
                if pval is not None and abs(pval - min_p) < 1e-9 else "")
    p.detail = (f"prospective: {basis}, effect={eff:+.2f}{spread} eV "
                f"(decision threshold {effect_size_threshold}), "
                f"p={pval if pval is None else round(pval, 4)}{at_min_p}, sig={sig}, "
                f"n={len(match)}/{len(other)} mols, "
                f"n_eff={len(match_units)}/{len(other_units)} series")
    return p
