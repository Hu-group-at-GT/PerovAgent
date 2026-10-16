import os

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.io import write
from ase.neighborlist import neighbor_list
from perovkit import BindingMotif, Core, Ligand, LigandSpec, Slab
from rdkit import Chem
from rdkit.Chem import AllChem

from perovagent.config import ConstructionConfig


def b_donor_cutoff(b_site: str, element: str) -> float:
    """Longest B-site--donor distance (Angstrom) that counts as coordination: 1.5x the sum of
    the two covalent radii (Cordero 2008, as ase tabulates them)."""
    return 1.5 * (covalent_radii[atomic_numbers[b_site]] + covalent_radii[atomic_numbers[element]])


class ConstructionTools:
    """Builds cores, ligands and covered slabs, saves them, and maps their geometry."""

    def __init__(self, cfg: ConstructionConfig, contact_cutoff: float):
        self.cfg = cfg
        self.contact_cutoff = contact_cutoff

    def build_ligand_conformers(self, smiles, seed):
        """Candidate geometries for the E_ligand reference, best-UFF-first."""
        # never relax more conformers than were embedded, and never fewer than one of each
        n_conformers = max(1, self.cfg.n_ligand_conformers)
        n_relax = max(1, min(self.cfg.n_ligand_relax, n_conformers))
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"[Construction] Conformer build failed: unparseable SMILES {smiles!r}")
                return None
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = int(seed)
            # ETKDG can return fewer conformers than requested (or none); handled, not asserted.
            cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
            if not cids:
                print(f"[Construction] Conformer build failed: no embedding for {smiles!r}")
                return None
            try:
                AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
            except Exception as e:  # noqa: BLE001
                print(f"[Construction] UFF conformer optimisation skipped: {e}")
            ranked = []
            for cid in cids:
                try:
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                    ranked.append((ff.CalcEnergy(), cid))
                except Exception:  # noqa: BLE001
                    ranked.append((float("inf"), cid))
            ranked.sort(key=lambda t: t[0])

            symbols = [a.GetSymbol() for a in mol.GetAtoms()]
            # The molecule's true bond table (RDKit bonds, not a geometry guess); relax_ligand
            # uses it to reject conformers that isomerised.
            ref_bonds = frozenset(
                tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds())
            out = []
            for _, cid in ranked[:n_relax]:
                out.append(Atoms(symbols=symbols, positions=mol.GetConformer(cid).GetPositions()))
            return out, ref_bonds
        except Exception as e:  # noqa: BLE001
            print(f"[Construction] Conformer build failed: {e}")
            return None

    def build_covered_slab(self, smiles, motif, name, seed):
        try:
            ligand = Ligand.from_smiles(
                smiles,
                BindingMotif(motif),
                random_seed=seed,
                name=name,
                optimize=False,
            )
        except Exception as e:
            print(f"[Construction] Ligand build failed: {e}")
            return None

        try:
            if self.cfg.binding_sites is None:
                raise ValueError(
                    "'binding_sites' must be provided in the construction config: it names the "
                    "exact\nX-site indices the ligands are placed on.")
            core = Core.build_slab(
                A=self.cfg.A, B=self.cfg.B, X=self.cfg.X,
                a=self.cfg.lattice_constant, supercell=self.cfg.supercell, vacuum=self.cfg.vacuum,
            )
            slab = Slab(core=core, ligand_specs=[
                LigandSpec(ligand=ligand, binding_sites=self.cfg.binding_sites["anionic"])])
            slab.place_ligands(overlap_cutoff=self.cfg.overlap_cutoff, random_seed=seed,
                               verbose=False)
        except Exception as e:
            print(f"[Construction] Slab build failed: {e}")
            return None

        return slab

    @staticmethod
    def save_slab(slab, save_dir, name):
        """Save the optimized slab structure to disk."""
        try:
            slab.to(fmt="vasp", filename=os.path.join(save_dir, f"opt_{name}.vasp"))
        except Exception as e:
            print(f"[Construction] Save slab failed: {e}")

    @staticmethod
    def save_ligand(atoms, save_dir, name):
        try:
            write(os.path.join(save_dir, f"{name}.xyz"), atoms, format="xyz")
        except Exception as e:
            print(f"[Construction] Save ligand failed: {e}")

    def contact_map(self, slab):
        """Relaxed-geometry contact map of each anionic ligand to the inorganic frame."""
        atoms = slab.atoms
        symbols = atoms.get_chemical_symbols()
        framework = {self.cfg.A, self.cfg.B, self.cfg.X}

        # Exclude every ligand atom from the framework pool so a ligand's own halide substituent
        # is never mistaken for a lattice halide.
        ligand_global = set()
        for lig in slab.ligands:
            if getattr(lig, "indices", None) is not None:
                ligand_global.update(int(i) for i in lig.indices)

        is_framework = np.array(
            [(symbols[i] in framework and i not in ligand_global) for i in range(len(atoms))],
            dtype=bool,
        )

        # global atom index -> (ligand id, local_index)
        src_of: dict = {}
        binding_of: dict = {}
        for lig in slab.ligands:
            idx = getattr(lig, "indices", None)
            if idx is None:
                continue
            lid = int(lig.id)
            binding_of[lid] = set(getattr(lig, "binding_atoms", None) or [])
            for k, g in enumerate(idx):
                src_of[int(g)] = (lid, k)

        try:
            I, J, Dist = neighbor_list("ijd", atoms, float(self.contact_cutoff))
        except Exception:
            return {}

        acc: dict = {}
        for a, b, dist in zip(I, J, Dist):
            a, b = int(a), int(b)
            src = src_of.get(a)
            if src is None:
                continue
            sym_a = symbols[a]
            if sym_a in ("C", "H"):
                continue
            if not is_framework[b]:
                continue
            lid, loc = src
            acc.setdefault((lid, loc, sym_a, loc in binding_of[lid]), []).append(
                {"element": symbols[b], "distance": round(float(dist), 2)})

        # assemble, heteroatoms in ligand-local-index order, contacts best-first by distance
        by_lig: dict = {}
        for key, contacts in acc.items():
            by_lig.setdefault(key[0], []).append((key, contacts))
        out: dict = {}
        for lid, items in by_lig.items():
            items.sort(key=lambda kc: kc[0][1])
            hets = []
            for (_lid, _loc, elem, is_donor_atom), contacts in items:
                contacts.sort(key=lambda c: c["distance"])
                hets.append({"element": elem, "is_donor_atom": is_donor_atom,
                             "contacts": contacts})
            out[lid] = hets
        return out


def infer_bonds(atoms, scale: float = 1.2) -> frozenset:
    """Bond set inferred from geometry. Detects a relaxation that changed the molecule rather
    than its conformation."""
    pos = atoms.get_positions()
    rc = np.array([covalent_radii[atomic_numbers[s]] for s in atoms.get_chemical_symbols()])
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    thr = scale * (rc[:, None] + rc[None, :])
    iu = np.triu_indices(len(pos), k=1)
    return frozenset((int(i), int(j)) for i, j in zip(*iu) if d[i, j] < thr[i, j])


def infer_bonds_pbc(atoms, scale: float = 1.2) -> frozenset:
    rc = np.array([covalent_radii[atomic_numbers[s]] for s in atoms.get_chemical_symbols()])
    i_idx, j_idx, dist = neighbor_list("ijd", atoms, float(scale * 2.0 * rc.max()))
    return frozenset((int(min(i, j)), int(max(i, j)))
                     for i, j, d in zip(i_idx, j_idx, dist) if d < scale * (rc[i] + rc[j]))


def _wl_certificate(symbols, bonds) -> tuple:
    """Weisfeiler-Lehman isomorphism test of an element-labelled graph."""
    adj: dict[int, list[int]] = {i: [] for i in range(len(symbols))}
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    lab = {i: symbols[i] for i in range(len(symbols))}
    for _ in range(3):  # WL refinement rounds
        lab = {i: lab[i] + "(" + ",".join(sorted(lab[k] for k in adj[i])) + ")" for i in adj}
    return tuple(sorted(lab.values()))


def ligand_graph_intact(slab, smiles: str) -> tuple[bool, str]:
    ligands = getattr(slab, "ligands", None)
    if not ligands:
        return True, ""
    lig = ligands[-1]
    idx = getattr(lig, "indices", None)
    if idx is None:
        return True, ""
    ref = Chem.AddHs(Chem.MolFromSmiles(smiles))

    sub = slab.atoms[idx]
    got_bonds = infer_bonds_pbc(sub)          # scale=1.2, calibrated
    got = _wl_certificate(sub.get_chemical_symbols(), got_bonds)
    want = _wl_certificate([a.GetSymbol() for a in ref.GetAtoms()],
                           [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in ref.GetBonds()])
    if got == want:
        return True, ""
    return False, (f"relaxed ligand has {len(got_bonds)} bonds vs {ref.GetNumBonds()} in "
                   f"{smiles}; connectivity changed")


def format_contact_map(contact_map: dict) -> str:
    """Compact description of the organic-inorganic contact map.
    e.g. L2{S*: Pb@2.86} L3{S*: Pb@2.86, Cs@3.41}."""
    groups = []
    for lig_id in sorted(contact_map, key=str):
        segs = []
        for h in contact_map[lig_id]:
            label = f"{h['element']}{'*' if h['is_donor_atom'] else ''}"
            contacts = [f"{c['element']}@{c['distance']}" for c in h["contacts"]]
            segs.append(f"{label}: {', '.join(contacts)}")
        groups.append(f"L{lig_id}{{{'; '.join(segs)}}}")
    return " ".join(groups)


def _check_test_ligand_is_placed(cfg: ConstructionConfig) -> None:
    """Refuse a construction config that would never place the molecule under test."""
    if cfg.binding_sites is not None and not cfg.binding_sites.get("anionic"):
        raise ValueError(
            "construction.binding_sites.anionic is empty: the candidate has no placement site, "
            "so all candidates would score identically.")


def describe_slab_system(cfg: ConstructionConfig) -> str:
    """Prompt description of the evaluated slab system."""
    A, B, X = cfg.A, cfg.B, cfg.X
    sc_str = "x".join(str(v) for v in cfg.supercell)
    n_surface = cfg.supercell[0] * cfg.supercell[1]

    lines = [
        "Evaluation system:",
        f"- Surface: {A}{B}{X}3 (001), {sc_str} supercell, cubic a={cfg.lattice_constant} A, "
        f"periodic in-plane, vacuum along z; only the top (+z) face is exposed.",
        f"- Top-face sites: {n_surface} X-sites ({X}) and {n_surface} A-sites ({A}).",
    ]
    if cfg.binding_sites is not None:
        n_lig = len(cfg.binding_sites["anionic"])
        lines.append(
            f"- Candidate coverage: {n_lig} X-site(s) of {n_surface} = "
            f"{100.0 * n_lig / n_surface:.0f}% X-site coverage; the ligand occupies "
            f"surface {X} sites and caps exposed, undercoordinated {B}.")
        lines.append(
            f"- Co-passivation: none; A-sites retain native {A}. Adsorbates are the candidate "
            f"and its periodic copies.")
    else:
        lines.append(
            "- Placement: `binding_sites` is absent, so the placement scheme is unspecified.")

    return "\n".join(lines)
