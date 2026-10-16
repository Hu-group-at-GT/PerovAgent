from __future__ import annotations

from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AdmissionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_heavy_atoms: int = 3
    max_heavy_atoms: int = 50
    max_mol_weight: Optional[float] = None
    allowed_atom_symbols: tuple[str, ...] = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I")
    reactive_smarts_blocklist: tuple[str, ...] = (
        "[CX3](=O)[F,Cl,Br,I]",         # acyl halide
        "[OX2][OX2]",                   # peroxide
        "[NX2]=[NX2]",                  # azo / diazo-ish
        "[N+]#[C-]",                    # isocyanide
        "[C]=[C]=[C]",                  # allene
    )

    @field_validator("min_heavy_atoms", "max_heavy_atoms")
    @classmethod
    def _nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be non-negative")
        return v


class ConstructionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    A: str = "Cs"
    B: str = "Pb"
    X: str = "Br"
    lattice_constant: float = 5.95
    supercell: tuple[int, int, int] = (2, 2, 3)
    vacuum: float = 20.0
    binding_sites: Optional[dict[str, Any]] = None
    overlap_cutoff: float = 1.6
    n_ligand_conformers: int = 200
    n_ligand_relax: int = 32

    @property
    def material(self) -> str:
        return f"{self.A}{self.B}{self.X}3"


class D3Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # D3 PBE parameters
    a1: float = 0.4289
    a2: float = 4.4407
    s8: float = 0.7875
    cutoff: float = 20.0


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fmax: float = 0.025
    max_steps: int = 1000


class RelaxerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: str = "cuda"
    checkpoint: str = "mattersim-v1.0.0-5m"
    d3: D3Config = Field(default_factory=D3Config)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)


class BindingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_placements: int = 5
    min_placements_ok: int = 3
    max_placement_std: float = 1.0
    use_reference_cache: bool = False
    e_vacancy_cached: Optional[float] = None
    max_binding_energy: float = -6.0

    @model_validator(mode="after")
    def _validate_e_vacancy_cache(self):
        if self.use_reference_cache and self.e_vacancy_cached is None:
            raise ValueError(
                "binding.use_reference_cache is True but binding.e_vacancy_cached is unset. "
                "Provide E_vacancy for this slab and calculator, or set use_reference_cache: "
                "False to relax the vacancy per placement.")
        return self


class CampaignConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "gpt-5.6-sol"
    reasoning_effort: Optional[str] = None
    effect_size_threshold: float
    max_retry: int = 3
    retrospective_min_per_group: int = 5
    prospective_min_per_group: int = 5
    min_anchor_stratum: int = 3
    significance_alpha: float = 0.1


class DiscoveryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campaign: CampaignConfig
    admission: AdmissionConfig = Field(default_factory=AdmissionConfig)
    construction: ConstructionConfig = Field(default_factory=ConstructionConfig)
    relaxer: RelaxerConfig = Field(default_factory=RelaxerConfig)
    binding: BindingConfig = Field(default_factory=BindingConfig)


def load_config(path: str) -> DiscoveryConfig:
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    return DiscoveryConfig.model_validate(raw)
