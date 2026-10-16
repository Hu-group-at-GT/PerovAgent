from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field
from rdkit import Chem


class AnchorSpec(BaseModel):
    """A machine-checkable description of an X-site binding head: the motif SMARTS,
    1-2 coordinating donor atoms, the binding mode (anionic/neutral), and the
    deprotonation site that reaches the bound -1 form (anionic only)."""
    name: str
    binding_mode: Literal["anionic", "neutral"]
    motif_smarts: str
    coord_atoms: list[str] = Field(default_factory=list)
    deprot_smarts: Optional[str] = None
    charge_target_index: Optional[int] = None

    @property
    def net_charge(self) -> int:
        return -1 if self.binding_mode == "anionic" else 0


class AnchorRegistry:
    """Anchors the agent has invented and the harness admitted. Born empty: there is no
    seed table, every entry arrives at runtime through `register`."""

    def __init__(self):
        self._specs: dict[str, AnchorSpec] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def get(self, name: str) -> Optional[AnchorSpec]:
        return self._specs.get(name)

    def specs(self) -> list[AnchorSpec]:
        return list(self._specs.values())

    def register(self, spec: AnchorSpec, probe_smiles: str, *,
                 allowed_elements) -> tuple[bool, str]:
        reason = validate_spec(spec, probe_smiles, allowed_elements=allowed_elements)
        if reason is not None:
            return False, reason
        self._specs[spec.name] = spec
        return True, ""


# Validation of anchor specs
def validate_spec(spec: AnchorSpec, probe_smiles: str, *,
                  allowed_elements) -> Optional[str]:
    """Return None if ``spec`` is well-formed, else a short reason."""
    if not spec.name.strip():
        return "empty anchor name"
    if Chem.MolFromSmarts(spec.motif_smarts) is None:
        return f"motif_smarts does not compile: {spec.motif_smarts!r}"
    n = len(spec.coord_atoms)
    if not (1 <= n <= 2):   # Perovkit's slab placement rejects binding motifs with >2 atoms
        return f"coord_atoms must have 1-2 entries, got {n}"
    donors = set(allowed_elements) - {"C", "H"}
    for el in spec.coord_atoms:
        if el not in donors:
            return f"coord atom {el!r} is not an accepted donor element {sorted(donors)}"
    if spec.binding_mode == "anionic":
        if not spec.deprot_smarts:
            return "anionic anchor requires deprot_smarts"
        patt = Chem.MolFromSmarts(spec.deprot_smarts)
        if patt is None:
            return f"deprot_smarts does not compile: {spec.deprot_smarts!r}"
        if spec.charge_target_index is None or spec.charge_target_index < 0:
            return "anionic anchor requires a non-negative charge_target_index"
        if spec.charge_target_index >= patt.GetNumAtoms():
            return (f"charge_target_index {spec.charge_target_index} out of range for a "
                    f"{patt.GetNumAtoms()}-atom deprot_smarts")
    else:
        if spec.deprot_smarts or spec.charge_target_index is not None:
            return "neutral anchor must not carry deprotonation fields"

    return _validate_against_molecule(spec, probe_smiles)


def _validate_against_molecule(spec: AnchorSpec, smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"probe smiles does not parse: {smiles!r}"
    matches = find_anchor_matches(mol, spec)
    if len(matches) == 0:
        return f"anchor motif {spec.motif_smarts!r} not found in {smiles!r}"
    if len(matches) > 1:
        return f"anchor motif {spec.motif_smarts!r} matches {len(matches)}x in {smiles!r} (need 1)"
    motif_syms = {mol.GetAtomWithIdx(i).GetSymbol() for i in matches[0]}
    for el in spec.coord_atoms:
        if el not in motif_syms:
            return (f"coord atom {el!r} not present among motif atoms {sorted(motif_syms)} "
                    f"of {smiles!r}")
    bound = to_bound_form(smiles, spec)
    if bound is None:
        return f"could not build the bound form of {smiles!r} for anchor {spec.name!r}"
    bmol = Chem.MolFromSmiles(bound)
    if bmol is None:
        return f"bound form {bound!r} does not parse"
    if Chem.GetFormalCharge(bmol) != spec.net_charge:
        return (f"bound form of {smiles!r} has net charge "
                f"{Chem.GetFormalCharge(bmol)}, expected {spec.net_charge}")
    return None


# Anchor-motif matching
def find_anchor_matches(mol: Chem.Mol, spec: AnchorSpec) -> tuple[tuple[int, ...], ...]:
    """All SMARTS matches for the anchor motif."""
    return mol.GetSubstructMatches(Chem.MolFromSmarts(spec.motif_smarts))


def to_bound_form(smiles: str, spec: AnchorSpec) -> Optional[str]:
    """Convert a SMILES string to its bound form based on the anchor.
    Anionic: deprotonate the acidic atom to the -1 bound form.
    Neutral: return the canonical SMILES."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if spec.binding_mode == "neutral":
        return Chem.MolToSmiles(mol, canonical=True)

    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(spec.deprot_smarts))
    if not matches:
        return None
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(matches[0][spec.charge_target_index])

    # Remove exactly one proton.
    n_h = atom.GetTotalNumHs()
    if n_h < 1:
        return None
    atom.SetFormalCharge(-1)
    atom.SetNumExplicitHs(n_h - 1)
    atom.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    out_mol = rw.GetMol()
    if Chem.GetFormalCharge(out_mol) != spec.net_charge:
        return None
    return Chem.MolToSmiles(out_mol, canonical=True)
