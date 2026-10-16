from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from perovagent.chemistry.anchors import AnchorSpec, find_anchor_matches
from perovagent.config import AdmissionConfig


@dataclass
class AdmittedLigand:
    canonical_smiles: str
    anchor_type: str


def canonical_identity_from_mol(mol: Chem.Mol) -> Optional[str]:
    """Canonical, stereo-free SMILES for an already-sanitized Mol, or None if it cannot be built."""
    try:
        Chem.RemoveStereochemistry(mol)
    except Exception:
        pass
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def canonical_identity(smiles: str) -> str:
    """Canonical, stereo-free SMILES for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return ""
    return canonical_identity_from_mol(mol) or ""


# ligand admission
def check_admissibility(
    raw_mol: Chem.Mol, cfg: AdmissionConfig, *, charge_window: tuple[int, int],
) -> tuple[Optional[Chem.Mol], Optional[str], Optional[str]]:
    """Enforce connectivity, size, MW, allowed elements, radicals, and a formal-charge window."""
    mol = Chem.Mol(raw_mol)  # deep copy
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None, None, "sanitize_error"

    canonical = canonical_identity_from_mol(mol)
    if canonical is None:
        return None, None, "sanitize_error"

    if "." in canonical:
        return None, None, "disconnected"

    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < cfg.min_heavy_atoms:
        return None, None, "too_small"
    if n_heavy > cfg.max_heavy_atoms:
        return None, None, "too_large"
    if cfg.max_mol_weight is not None and Descriptors.MolWt(mol) > cfg.max_mol_weight:
        return None, None, "mol_weight_too_high"

    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym != "H" and sym not in cfg.allowed_atom_symbols:
            return None, None, f"forbidden_element:{sym}"

    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            return None, None, "has_radical"

    lo, hi = charge_window
    fc = Chem.GetFormalCharge(mol)
    if not (lo <= fc <= hi):
        return None, None, f"charge_violation_for_anchor:{fc}"

    return mol, canonical, None


def admit_ligand(raw_mol: Chem.Mol, spec: AnchorSpec,
                 cfg: AdmissionConfig) -> tuple[Optional[AdmittedLigand], Optional[str]]:
    mol, canonical, reason = check_admissibility(
        raw_mol, cfg, charge_window=(spec.net_charge, 0))
    if mol is None:
        return None, reason
    if not find_anchor_matches(mol, spec):
        return None, "missing_anchor_motif"
    return AdmittedLigand(canonical_smiles=canonical, anchor_type=spec.name), None


# reactive substructure gate
def molecule_chemistry_gate(mol: Chem.Mol, cfg: AdmissionConfig) -> Optional[str]:
    for smarts in cfg.reactive_smarts_blocklist:
        patt = Chem.MolFromSmarts(smarts)          # a SMARTS that fails to compile is skipped
        if patt is not None and mol.HasSubstructMatch(patt):
            return f"reactive_substructure:{smarts}"
    return None


# scaffold labelling
def murcko_key(smiles: str) -> str:
    """A molecule's Murcko scaffold as canonical SMILES ("<acyclic>" without a ring)."""
    try:
        s = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smiles)))
    except Exception:
        s = ""
    return s or "<acyclic>"


def describe_admission_constraints(cfg: AdmissionConfig) -> str:
    lines = [
        "Admission constraints (checked before evaluation): a molecule that violates any "
        "of them is rejected without ever being measured, so proposing outside them "
        "spends the round for nothing.",
        f"- Allowed elements: {', '.join(cfg.allowed_atom_symbols)}, plus H.",
        f"- Size: {cfg.min_heavy_atoms}-{cfg.max_heavy_atoms} heavy atoms.",
    ]
    if cfg.max_mol_weight is not None:
        lines.append(f"- Molecular weight <= {cfg.max_mol_weight:g}.")
    lines.append("- Chemistry: one connected molecule; no radicals; no reactive or unstable "
                 "substructures.")
    return "\n".join(lines)
