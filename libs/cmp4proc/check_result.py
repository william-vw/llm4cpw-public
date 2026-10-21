"""
check_result.py

Base classes for the results of the constraint checkers (ltl/lola/ltl_check.py,
activ_cnstr/activ_cnstr_check.py): each checker's check_constraints() returns a CheckResult subclass
holding one ConstraintResult subclass per checked constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConstraintResult:
    name: str
    holds: bool
    error: Optional[str] = field(default=None, kw_only=True)  # why the constraint could not be checked, if so


@dataclass
class CheckResult:
    results: List[ConstraintResult]
    pnml_path: Optional[str]  # None if PNML export was skipped

    def filter_results(self, holds: bool) -> List[ConstraintResult]:
        """The constraint results that hold (`holds=True`) or that do not (`holds=False`)."""
        return [result for result in self.results if result.holds == holds]

    def score(self) -> float:
        """The fraction of constraint results that hold; 1.0 if there are no constraints."""
        if not self.results:
            return 1.0
        return len(self.filter_results(True)) / len(self.results)

    def summary(self) -> str:
        """"FAILED: " followed by the comma-separated names of the failed constraints (each with its
        error in quotes, if it has one), or "SUCCESS" if all of them hold."""
        failed = self.filter_results(False)
        if not failed:
            return "SUCCESS"
        return "FAILED: " + ",".join(f.name for f in failed)
            # f.name + (f"(\"{f.error}\")" if f.error is not None else "") for f in failed)

    @property
    def all_hold(self) -> bool:
        return all(result.holds for result in self.results)
