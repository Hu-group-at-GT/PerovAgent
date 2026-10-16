# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from NVIDIA nvalchemi-toolkit-ops (examples/dispersion/utils.py) for perovagent.

# The DFT-D3 parameters used in the default example are imported and converted from
# from the Grimme group website (https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/)
# and processed into a PyTorch state dictionary with the expected structure for the
# `nvalchemiops` DFT-D3 kernels.

"""DFT-D3 dispersion calculator plus helpers to generate its parameter file from the
reference Grimme-group Fortran implementation."""

from __future__ import annotations

import io
import os
import re
import tarfile
from hashlib import md5
from pathlib import Path
from typing import Literal

import numpy as np
import requests
import torch
import torch.nn as nn

from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.interactions.dispersion import (
    dftd3,
    D3Parameters,
)

from ase.calculators.calculator import Calculator, all_changes

__all__ = [
    "DFTD3",
    "DFTD3Calculator",
    "extract_dftd3_parameters",
    "save_dftd3_parameters",
    "load_d3_parameters",
]

# URL for DFT-D3 parameter files from Grimme group
DFTD3_TGZ_URL = "https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/dftd3.tgz"
REFERENCE_MD5 = "a76c752e587422c239c99109547516d2"

# Unit conversion constants from CODATA 2022 (retrieved 2025-11-12)
# Bohr radius: 5.291 772 105 44 x 10^-11 m
# Hartree energy in eV: 27.211 386 245 981 eV
BOHR_TO_ANGSTROM = 0.529177210544
HARTREE_TO_EV = 27.211386245981
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM


class DFTD3(nn.Module):
    """DFT-D3(BJ) dispersion correction as a PyTorch module.

    a1, a2 (Becke-Johnson damping) and s8 (C8 scaling) are functional-specific.
    `units`: "atomic" (Bohr, Hartree) or "conventional" (Angstrom, eV, default).
    """

    def __init__(
        self,
        a1: float,
        a2: float,
        s8: float,
        units: Literal["atomic", "conventional"] = "conventional",
        k1: float = 16.0,
        k3: float = -4.0,
        s6: float = 1.0,
        s5_smoothing_on: float = 1e10,
        s5_smoothing_off: float = 1e10,
    ):
        super().__init__()

        # Validate units
        if units not in ("atomic", "conventional"):
            raise ValueError(f"units must be 'atomic' or 'conventional', got {units}")

        # Store dimensionless parameters
        self.a1 = a1
        self.a2 = a2
        self.s6 = s6
        self.s8 = s8
        self.k1 = k1
        self.k3 = k3
        self.units = units

        self.s5_smoothing_on = s5_smoothing_on
        self.s5_smoothing_off = s5_smoothing_off
        if units == "conventional":
            self.s5_smoothing_on *= ANGSTROM_TO_BOHR
            self.s5_smoothing_off *= ANGSTROM_TO_BOHR

        # Load DFT-D3 parameters from cache directory
        cache_dir = Path(os.path.expanduser("~")) / ".cache" / "nvalchemiops"
        param_file = cache_dir / "dftd3_parameters.pt"

        if not param_file.exists():
            raise FileNotFoundError(
                f"DFT-D3 parameter file not found: {param_file}\n"
                "Generate it once with: python -m perovagent.evaluation.dftd3"
            )

        # Load parameters as D3Parameters instance
        d3_params = load_d3_parameters(param_file)

        # Register parameters as buffers (non-trainable, part of module state)
        # All parameters are in atomic units
        self.register_buffer(
            "covalent_radii",
            d3_params.rcov.float(),
            persistent=True,
        )
        self.register_buffer(
            "r4r2",
            d3_params.r4r2.float(),
            persistent=True,
        )
        self.register_buffer(
            "c6_reference",
            d3_params.c6ab.float(),
            persistent=True,
        )
        self.register_buffer(
            "coord_num_ref",
            d3_params.cn_ref.float(),
            persistent=True,
        )

    def forward(
        self,
        positions: torch.Tensor,
        numbers: torch.Tensor,
        neighbor_matrix: torch.Tensor,
        batch_idx: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
        neighbor_matrix_shifts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute DFT-D3 dispersion energy and forces.

        positions in units set by self.units; numbers int32; neighbor_matrix
        [num_atoms, max_neighbors] int32 with padding entries >= num_atoms.
        Returns (energy, forces, coord_num).
        """
        # Detach all input tensors to prevent backpropagation through inputs
        positions = positions.detach()
        if cell is not None:
            cell = cell.detach()
        if neighbor_matrix_shifts is not None:
            neighbor_matrix_shifts = neighbor_matrix_shifts.detach()

        # Convert inputs to atomic units if needed
        if self.units == "conventional":
            positions = positions * ANGSTROM_TO_BOHR
            if cell is not None:
                cell = cell * ANGSTROM_TO_BOHR

        # Call the underlying compute function
        energy, forces, coord_num = dftd3(
            positions=positions,
            numbers=numbers,
            neighbor_matrix=neighbor_matrix,
            covalent_radii=self.covalent_radii,
            r4r2=self.r4r2,
            c6_reference=self.c6_reference,
            coord_num_ref=self.coord_num_ref,
            a1=self.a1,
            a2=self.a2,
            s8=self.s8,
            k1=self.k1,
            k3=self.k3,
            s6=self.s6,
            s5_smoothing_on=self.s5_smoothing_on,
            s5_smoothing_off=self.s5_smoothing_off,
            batch_idx=batch_idx,
            cell=cell,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        # Convert outputs back to conventional units if needed
        if self.units == "conventional":
            # Energy: Hartree -> eV
            energy = energy * HARTREE_TO_EV
            # Forces: Hartree/Bohr -> eV/Angstrom
            forces = forces * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)

        return energy, forces, coord_num

    def extra_repr(self) -> str:
        return f"a1={self.a1}, a2={self.a2:.4f}, s8={self.s8}, units={self.units}"


def _download_and_extract_tgz(url: str) -> dict[str, str]:
    """Download a .tgz archive and return {basename: contents} for its .f/.F Fortran files.
    Verifies the MD5 checksum; raises on download/extraction/checksum failure."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Verify MD5 checksum
    content_bytes = response.content
    hasher = md5(usedforsecurity=False)
    hasher.update(content_bytes)
    computed_md5 = hasher.hexdigest()
    if computed_md5 != REFERENCE_MD5:
        raise ValueError(
            f"MD5 checksum verification failed for downloaded archive.\n"
            f"Expected: {REFERENCE_MD5}\n"
            f"Got:      {computed_md5}\n"
            "The archive may have been modified or corrupted. "
            "Please verify the source or provide a local dftd3_ref directory path."
        )

    # Extract files from tar.gz archive
    extracted_files = {}
    with tarfile.open(fileobj=io.BytesIO(content_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith((".f", ".F")):
                # Extract file content
                file_obj = tar.extractfile(member)
                if file_obj is not None:
                    content = file_obj.read().decode("utf-8", errors="ignore")
                    # Store with base filename as key
                    basename = Path(member.name).name
                    extracted_files[basename] = content

    return extracted_files


def _find_fortran_array(content: str, var_name: str) -> np.ndarray:
    """Parse a Fortran `data` array by variable name (skipping comments) into a float64 array."""
    lines = content.splitlines()

    in_data_block = False
    data_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("!") or stripped.lower().startswith("c "):
            continue

        if not in_data_block:
            if re.match(rf"^\s*data\s+{var_name}\s*/\s*", line, re.IGNORECASE):
                in_data_block = True
                data_lines.append(line)
        else:
            data_lines.append(line)
            if "/" in line and not line.strip().startswith("!"):
                break

    if not data_lines:
        raise ValueError(f"Variable '{var_name}' not found in Fortran source")

    data_str = " ".join(data_lines)
    pattern = rf"data\s+{var_name}\s*/\s*(.*?)\s*/"
    match = re.search(pattern, data_str, re.DOTALL | re.IGNORECASE)

    if not match:
        raise ValueError(f"Failed to parse '{var_name}'")

    content = match.group(1)
    lines_clean = []
    for line in content.split("\n"):
        if "!" in line:
            line = line[: line.index("!")]
        lines_clean.append(line)
    content = " ".join(lines_clean)

    numbers = re.findall(r"[-+]?\d+\.\d+(?:_wp)?", content)
    values = [float(n.replace("_wp", "")) for n in numbers]

    return np.array(values, dtype=np.float64)


def _parse_pars_array(content: str) -> np.ndarray:
    """Parse the pars array into an [n_records, 5] float64 array of
    [C6, Z_i, Z_j, CN_i, CN_j] records."""
    values = []
    in_data_section = False

    for line in content.splitlines():
        if "real*8" in line.lower() and "pars" in line.lower():
            continue

        if "pars(" in line.lower() and "=(" in line:
            in_data_section = True

        if not in_data_section:
            continue

        if "/)" in line:
            in_data_section = False

        if "!" in line:
            line = line[: line.index("!")]

        line = line.replace("pars(", " ").replace("=(/", " ")
        line = line.replace("/)", " ").replace(":", " ")

        numbers = re.findall(r"[-+]?\d+\.\d+[eEdD][-+]?\d+", line)
        values.extend(
            float(num_str.replace("D", "e").replace("d", "e")) for num_str in numbers
        )

    values = np.array(values, dtype=np.float64)
    n_records = len(values) // 5

    if len(values) % 5 != 0:
        values = values[: n_records * 5]

    return values.reshape(n_records, 5)


def _limit(encoded: int) -> tuple[int, int]:
    """Decode a Fortran element encoding (atomic_number + 100*(cn_index-1)) into
    (atom, cn_idx): atomic number 1-94 and coordination-number index 1-5."""
    atom = encoded
    cn_idx = 1

    while atom > 100:
        atom -= 100
        cn_idx += 1

    return atom, cn_idx


def _build_arrays(pars_records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build c6ab and cn_ref arrays [95, 95, 5, 5] float32 from pars records.
    Index 0 is unused (padding); valid atomic numbers are 1-94."""
    c6ab = np.zeros((95, 95, 5, 5), dtype=np.float32)
    cn_ref = np.full((95, 95, 5, 5), -1.0, dtype=np.float32)
    cn_values = {elem: {} for elem in range(95)}

    for record in pars_records:
        c6_val, z_i_enc, z_j_enc, cn_i, cn_j = record

        iat, iadr = _limit(int(z_i_enc))
        jat, jadr = _limit(int(z_j_enc))

        if iat < 1 or iat > 94 or jat < 1 or jat > 94:
            continue
        if iadr < 1 or iadr > 5 or jadr < 1 or jadr > 5:
            continue

        iadr_py = iadr - 1
        jadr_py = jadr - 1

        c6ab[iat, jat, iadr_py, jadr_py] = c6_val
        c6ab[jat, iat, jadr_py, iadr_py] = c6_val

        if iadr_py not in cn_values[iat]:
            cn_values[iat][iadr_py] = cn_i
        if jadr_py not in cn_values[jat]:
            cn_values[jat][jadr_py] = cn_j

    for elem in range(1, 95):
        for partner in range(1, 95):
            for cn_idx in range(5):
                if cn_idx in cn_values[elem]:
                    cn_ref[elem, partner, cn_idx, :] = cn_values[elem][cn_idx]

    return c6ab, cn_ref


def extract_dftd3_parameters(
    dftd3_ref_dir: Path | None = None,
) -> dict[str, torch.Tensor]:
    """Extract DFT-D3 parameters from local Fortran files (dftd3_ref_dir) or the Grimme download.

    Returns {"rcov", "r4r2", "c6ab", "cn_ref"} tensors in atomic units (Bohr for distances);
    index 0 is padding, valid atomic numbers are 1-94.
    """
    if dftd3_ref_dir is not None:
        # Use local files
        if not dftd3_ref_dir.exists():
            raise FileNotFoundError(f"Directory not found: {dftd3_ref_dir}")

        dftd3_f = dftd3_ref_dir / "dftd3.f"
        pars_f = dftd3_ref_dir / "pars.f"

        if not dftd3_f.exists():
            raise FileNotFoundError(f"File not found: {dftd3_f}")
        if not pars_f.exists():
            raise FileNotFoundError(f"File not found: {pars_f}")

        print(f"Reading DFT-D3 parameter files from: {dftd3_ref_dir}")
        with open(dftd3_f) as f:
            dftd3_content = f.read()
        with open(pars_f) as f:
            pars_content = f.read()
        print("  ✓ Files loaded")
    else:
        # Download from web
        print("Downloading DFT-D3 parameter files from Grimme group website...")
        try:
            print(f"  Downloading archive from {DFTD3_TGZ_URL}")
            extracted_files = _download_and_extract_tgz(DFTD3_TGZ_URL)
            print("  ✓ Download and extraction complete")

            # Extract the required files
            if "dftd3.f" not in extracted_files:
                raise ValueError("dftd3.f not found in archive")
            if "pars.f" not in extracted_files:
                raise ValueError("pars.f not found in archive")

            dftd3_content = extracted_files["dftd3.f"]
            pars_content = extracted_files["pars.f"]

        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to download DFT-D3 parameter files: {e}\n"
                "Please check your internet connection and try again, or provide "
                "a local dftd3_ref directory path."
            ) from e
        except (tarfile.TarError, ValueError) as e:
            raise RuntimeError(
                f"Failed to extract DFT-D3 parameter files from archive: {e}\n"
                "The archive format may have changed. Please provide "
                "a local dftd3_ref directory path."
            ) from e

    # Parse Fortran arrays
    print("Parsing Fortran source files...")
    r2r4_94 = _find_fortran_array(dftd3_content, "r2r4")
    rcov_94 = _find_fortran_array(dftd3_content, "rcov")
    pars_records = _parse_pars_array(pars_content)

    # Build parameter arrays (index 0 reserved, elements 1-94)
    r4r2 = np.zeros(95, dtype=np.float32)
    r4r2[1:95] = r2r4_94.astype(np.float32)

    rcov = np.zeros(95, dtype=np.float32)
    rcov[1:95] = rcov_94.astype(np.float32)

    c6ab, cn_ref = _build_arrays(pars_records)
    print("  ✓ Parsing complete")

    # Convert to PyTorch tensors
    return {
        "rcov": torch.from_numpy(rcov),
        "r4r2": torch.from_numpy(r4r2),
        "c6ab": torch.from_numpy(c6ab),
        "cn_ref": torch.from_numpy(cn_ref),
    }


def save_dftd3_parameters(parameters: dict[str, torch.Tensor]) -> Path:
    """Save the parameter dict to ~/.cache/nvalchemiops/dftd3_parameters.pt; returns the path."""
    cache_dir = Path(os.path.expanduser("~")) / ".cache" / "nvalchemiops"
    cache_dir.mkdir(parents=True, exist_ok=True)

    param_file = cache_dir / "dftd3_parameters.pt"
    torch.save(parameters, param_file)

    return param_file


def load_d3_parameters(
    param_file: Path | None = None,
) -> D3Parameters:
    """Load DFT-D3 parameters as a D3Parameters instance (default: the cached parameter file)."""
    if param_file is None:
        cache_dir = Path(os.path.expanduser("~")) / ".cache" / "nvalchemiops"
        param_file = cache_dir / "dftd3_parameters.pt"

    if not param_file.exists():
        raise FileNotFoundError(
            f"DFT-D3 parameter file not found: {param_file}\n"
            "Generate it once with: python -m perovagent.evaluation.dftd3"
        )

    state_dict = torch.load(param_file, map_location="cpu", weights_only=True)

    return D3Parameters(
        rcov=state_dict["rcov"],
        r4r2=state_dict["r4r2"],
        c6ab=state_dict["c6ab"],
        cn_ref=state_dict["cn_ref"],
    )

class DFTD3Calculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        a1: float,
        a2: float,
        s8: float,
        cutoff: float = 20.0, 
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        method: str = "naive",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cutoff = float(cutoff)
        self.device = torch.device(device)
        self.dtype = dtype
        self.method = method

        self.d3 = DFTD3(
            a1=a1, a2=a2, s8=s8,
            units="conventional",
        ).to(self.device)
        self.d3.eval()

    @torch.no_grad()
    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        atoms = self.atoms

        positions = torch.as_tensor(
            np.asarray(atoms.get_positions(), dtype=np.float32),
            device=self.device, dtype=self.dtype
        )
        numbers = torch.as_tensor(
            np.asarray(atoms.get_atomic_numbers(), dtype=np.int32),
            device=self.device, dtype=torch.int32
        )
        cell = torch.as_tensor(
            np.asarray(atoms.get_cell().array, dtype=np.float32),
            device=self.device, dtype=self.dtype
        )
        pbc = torch.as_tensor(
            np.asarray(atoms.get_pbc(), dtype=bool),
            device=self.device, dtype=torch.bool
        )

        neighbor_matrix, num_neighbors, shifts = neighbor_list(
            positions, cutoff=self.cutoff, cell=cell, pbc=pbc, method=self.method
        )

        energy, forces, _cn = self.d3(
            positions=positions,
            numbers=numbers,
            cell=cell,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=shifts,
        )

        self.results["energy"] = float(energy.detach().cpu().numpy().reshape(-1)[0])
        self.results["forces"] = forces.detach().cpu().numpy().astype(np.float64)


if __name__ == "__main__":
    params = extract_dftd3_parameters()
    save_dftd3_parameters(params)