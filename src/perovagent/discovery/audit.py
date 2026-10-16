from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit import Chem

from perovagent.discovery.principles import (DesignPrinciple, _fmt, _signed_effect,
                                             _split_by_predicate)


# anchor families the anchor-confound measurement stratifies by
_ANCHOR_FAMILY_PATTERNS = tuple((name, Chem.MolFromSmarts(sma)) for name, sma in (
    ("P_oxoacid",       "[PX4](=[OX1])[OX2H1,OX1-]"),
    ("P_thioacid",      "[PX4](=[SX1])[OX2H1,OX1-,SX2H1,SX1-]"),
    ("phosphine_oxide", "[PX4](=[OX1])"),
    ("phosphine",       "[PX3]"),
    ("S_oxoacid",       "[SX4,SX3](=[OX1])[OX2H1,OX1-]"),
    ("sulfate_ester",   "[SX4](=[OX1])(=[OX1])([OX2])[OX2]"),
    ("boronic",         "[BX3]([OX2H1,OX1-])[OX2H1,OX1-]"),
    ("tetrazole",       "[nX2,nX3H1]1[nX2][nX2][nX2,nX3H1][cX3]1"),
    ("hydroxamic",      "[CX3](=[OX1])[NX3][OX2H1,OX1-]"),
    ("carboxyl",        "[CX3](=[OX1])[OX2H1,OX1-]"),
    ("thiol",           "[SX2H1,SX1-]"),
    ("sulfoxide",       "[SX3](=[OX1])"),
    ("thioether",       "[SX2]([#6])[#6]"),
    ("nitrile",         "[NX1]#[CX2]"),
    ("N_aromatic",      "[nX2]"),
    ("amide",           "[CX3](=[OX1])[NX3]"),
    ("amine",           "[NX3;!$(N[CX3]=[OX1,SX1]);!$(N[SX4]);!$(N[PX4])]"),
    ("alcohol",         "[OX2H1][#6]"),
    ("ether",           "[OX2]([#6])[#6]"),
))


def anchor_family(smiles: str, fallback: str) -> str:
    """Coarse anchor family of `smiles`, or `fallback` when no family pattern matches.

    Falling back to the anchor name keeps an unrecognised anchor in its own stratum rather than
    merging chemistries that were never shown to be alike.
    """
    mol = Chem.MolFromSmiles(smiles)
    for name, patt in _ANCHOR_FAMILY_PATTERNS:
        hit = mol.GetSubstructMatch(patt)
        if not hit:
            continue
        if name not in ("P_oxoacid", "P_thioacid", "S_oxoacid"):
            return name
        centre = mol.GetAtomWithIdx(hit[0])
        n_het = sum(1 for a in centre.GetNeighbors() if a.GetSymbol() in ("O", "S"))
        n_c = sum(1 for a in centre.GetNeighbors() if a.GetSymbol() == "C")
        return f"{name}_{n_het}O{n_c}C"
    return fallback


# confound measurements for a principle
def _stratified_effect(principle, groups_m, groups_o, min_stratum):
    """Size-weighted mean signed effect over strata where both groups have >= min_stratum
    members. Returns (n_usable_strata, weighted_effect or None)."""
    usable = [k for k in set(groups_m) | set(groups_o)
              if len(groups_m.get(k, [])) >= min_stratum and len(groups_o.get(k, [])) >= min_stratum]
    if not usable:
        return 0, None
    num, den = 0.0, 0
    for k in usable:
        w = len(groups_m[k]) + len(groups_o[k])
        num += w * _signed_effect(principle, groups_m[k], groups_o[k])
        den += w
    return len(usable), num / den


def audit_principle(p: DesignPrinciple, records, *, min_stratum: int) -> DesignPrinciple:
    """Measure degenerate partitions, anchor identity, and molecular size as potential confounds,
    and record them on `p`'s audit fields (full precision) and `audit_notes`.

    The audit reports measurements without changing the verdict.
    """
    match_recs, other_recs = _split_by_predicate(p, records)
    n_m, n_o = len(match_recs), len(other_recs)
    total = n_m + n_o
    p.frac_match = (n_m / total) if total else 0.0
    p.degenerate_partition = (p.frac_match < 0.1 or p.frac_match > 0.9
                              or min(n_m, n_o) < min_stratum)
    # cleared first: a degenerate partition measures none of them
    p.n_usable_anchor_strata, p.within_anchor_effect = 0, None
    p.anchor_perfectly_collinear, p.within_size_effect = False, None
    if p.degenerate_partition:
        p.audit_notes = (f"degenerate partition n={n_m}/{n_o} (frac_match={p.frac_match:.2f}); "
                         "predicate matches ~all or ~none of the pool")
        return p

    effect_all = _signed_effect(p, [r.binding_energy for r in match_recs],
                                [r.binding_energy for r in other_recs])

    match_by_anchor, other_by_anchor = defaultdict(list), defaultdict(list)
    for r in match_recs:
        match_by_anchor[anchor_family(r.smiles, r.anchor)].append(r.binding_energy)
    for r in other_recs:
        other_by_anchor[anchor_family(r.smiles, r.anchor)].append(r.binding_energy)
    p.n_usable_anchor_strata, p.within_anchor_effect = _stratified_effect(
        p, match_by_anchor, other_by_anchor, min_stratum)
    # perfectly collinear = every anchor family is pure-match or pure-non-match (untestable within)
    p.anchor_perfectly_collinear = not any(
        match_by_anchor.get(a) and other_by_anchor.get(a)
        for a in set(match_by_anchor) | set(other_by_anchor))
    m_heavy = [(r, Chem.MolFromSmiles(r.smiles).GetNumHeavyAtoms()) for r in match_recs]
    o_heavy = [(r, Chem.MolFromSmiles(r.smiles).GetNumHeavyAtoms()) for r in other_recs]
    med = float(np.median([h for _, h in m_heavy + o_heavy]))
    match_by_size, other_by_size = defaultdict(list), defaultdict(list)
    for by_size, rows in ((match_by_size, m_heavy), (other_by_size, o_heavy)):
        for r, h in rows:
            by_size["small" if h <= med else "large"].append(r.binding_energy)
    _, p.within_size_effect = _stratified_effect(p, match_by_size, other_by_size, min_stratum)

    p.audit_notes = (f"n={n_m}/{n_o}, effect_all={_fmt(effect_all)}eV, "
                     f"usable_anchor_strata={p.n_usable_anchor_strata}, "
                     f"within_anchor={_fmt(p.within_anchor_effect)}, "
                     f"within_size={_fmt(p.within_size_effect)}"
                     + (" [anchor-collinear: no family has the predicate on both sides, so the "
                        "within-anchor contrast is unavailable; interpret this as an anchor-level "
                        "claim, not necessarily a confounded one]"
                        if p.anchor_perfectly_collinear else ""))
    return p
