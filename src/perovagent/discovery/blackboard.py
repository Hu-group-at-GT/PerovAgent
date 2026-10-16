from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

from perovagent.discovery.principles import DesignPrinciple
from perovagent.chemistry.admission import murcko_key


OBJECTIVE_LABEL = "binding energy (eV; STRONGEST = most negative = better)"


def rank_key(rec: CandidateRecord) -> float:
    """Sort key: the strongest (most negative) binder first."""
    return rec.binding_energy


def fmt_binding_energy(rec: CandidateRecord) -> str:
    return f"BE={rec.binding_energy:+.2f}"


@dataclass
class CandidateRecord:
    smiles: str
    anchor: str
    declared_anchor: str
    binding_energy: float
    binding_energy_std: float
    n_placements_ok: int
    n_placements_reacted: int
    geometry: str
    source: str
    murcko: str = field(init=False)
    rationale: str
    hypothesis: str
    predicted_direction: str
    parent_smiles: str

    def __post_init__(self):
        self.murcko = murcko_key(self.smiles)


class CandidatePool:
    """Every evaluated molecule, keyed by canonical SMILES."""
    def __init__(self):
        self._by_smiles: dict[str, CandidateRecord] = {}

    def add(self, rec: CandidateRecord) -> None:
        self._by_smiles[rec.smiles] = rec

    def all(self) -> list[CandidateRecord]:
        return list(self._by_smiles.values())

    def __len__(self) -> int:
        return len(self._by_smiles)

    def best(self, k: int) -> list[CandidateRecord]:
        return sorted(self._by_smiles.values(), key=rank_key)[:k]

    def best_value(self) -> Optional[float]:
        b = self.best(1)
        return b[0].binding_energy if b else None

    def coverage(self) -> dict[str, tuple[int, Optional[float]]]:
        """Coverage of the design space by Murcko scaffold: family -> (count, strongest binding
        energy), strongest family first."""
        agg: dict[str, list[CandidateRecord]] = {}
        for r in self._by_smiles.values():
            agg.setdefault(r.murcko, []).append(r)
        best = {k: min(recs, key=rank_key) for k, recs in agg.items()}
        return {k: (len(agg[k]), best[k].binding_energy)
                for k in sorted(agg, key=lambda k: rank_key(best[k]))}


class PrincipleLedger:
    """Proposed/validated design principles, keyed by principle_id."""
    def __init__(self):
        self._by_id: dict[str, DesignPrinciple] = {}

    def add(self, p: DesignPrinciple) -> None:
        self._by_id[p.principle_id] = p

    def get(self, principle_id: str) -> Optional[DesignPrinciple]:
        return self._by_id.get(principle_id)

    def all(self) -> list[DesignPrinciple]:
        return list(self._by_id.values())

    def supported(self) -> list[DesignPrinciple]:
        return [p for p in self._by_id.values() if p.status == "supported"]

    def candidates(self) -> list[DesignPrinciple]:
        return [p for p in self._by_id.values() if p.status == "candidate"]

    def conditionable(self) -> list[DesignPrinciple]:
        return [p for p in self._by_id.values() if p.status in ("supported", "candidate")]


@dataclass
class Budget:
    evals_cap: int
    iters_cap: int
    evals_spent: int = 0
    iters_spent: int = 0

    def can_continue(self) -> bool:
        return self.evals_spent < self.evals_cap and self.iters_spent < self.iters_cap

    def remaining_evals(self) -> int:
        return self.evals_cap - self.evals_spent


@dataclass
class DecisionLog:
    entries: list[dict] = field(default_factory=list)

    def append(self, iteration: int, decision, result_summary: str,
               convergence_stats: dict) -> None:
        experiment = decision.action == "EXPERIMENT"
        self.entries.append({
            "iter": iteration,
            "t": time.time(),
            "action": decision.action,
            "goal": decision.goal if experiment else None,
            "control": decision.control if experiment else None,
            "n": decision.n if experiment else None,
            "brief": decision.brief if experiment else "",
            "principle_id": (decision.principle_id or None) if experiment else None,
            "hypothesis": decision.hypothesis,
            "rationale": decision.rationale,
            "expected_outcome": decision.expected_outcome,
            "principle": (decision.principle.model_dump()
                          if decision.principle is not None else None),
            "result": result_summary,
            "convergence_stats": convergence_stats,
        })


@dataclass
class Blackboard:
    goal: str
    budget: Budget
    pool: CandidatePool = field(default_factory=CandidatePool)
    ledger: PrincipleLedger = field(default_factory=PrincipleLedger)
    log: DecisionLog = field(default_factory=DecisionLog)


# convergence metrics
def binding_energy_stats(recs) -> dict:
    """n, best (most negative), mean, median and spread of the measured binding energies of
    `recs` -- one batch, or the whole pool."""
    vals = [r.binding_energy for r in recs]
    if not vals:
        return {"n": 0, "best": None, "mean": None, "median": None, "spread": None}
    best = min(vals)
    return {"n": len(vals),
            "best": best,
            "mean": sum(vals) / len(vals),
            "median": statistics.median(vals),
            "spread": max(vals) - best}
