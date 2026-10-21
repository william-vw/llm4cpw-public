"""
activ_cnstr_check.py

Check the PartOf constraints of the constraints ontology (constraints.owl) against a BPMN model:

    BPMN -> PM4Py Petri net -> PNML file -> reloaded PNML -> transition labels
                                                                 |
    constraints.owl (rdflib) -> PartOf constraints --------------+--> PASS/FAIL per constraint

A PartOf constraint has one aggregate task (:hasAggregate) and one or more part tasks (:hasPart).
It holds when the aggregate task occurs as a (labeled) transition of the model and none of its
part tasks do -- the model has to stay at the level of the aggregate.

Library usage
-------------
from libs.cmp4proc.activ_cnstr.activ_cnstr_check import check_constraints

result = check_constraints("constraints.owl", "model.bpmn", "Heart Failure")
for r in result.results:
    print(r.name, r.holds, r.error)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote

import pm4py
from rdflib import Graph, Literal, Namespace, RDF, RDFS

from libs.cmp4proc.check_result import CheckResult, ConstraintResult

CNSTR = Namespace("http://notation3.org/constraints.owl#")


@dataclass
class ActivConstraintResult(ConstraintResult):
    aggregate: str  # label of the has-aggregate task
    parts: List[str]  # labels of the has-part tasks
    aggregate_found: bool  # the aggregate task occurs as a transition
    parts_found: List[str] = field(default_factory=list)  # has-part tasks that (wrongly) occur as transitions


@dataclass
class ActivCheckResult(CheckResult):
    activities: List[str]  # labels of the PNML's visible transitions


def _label(iri, g: Graph) -> str:
    label = g.value(iri, RDFS.label)
    return str(label) if label is not None else unquote(str(iri).rsplit("#", 1)[-1]).replace("_", " ")


def _natural_key(iri):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(iri))]


def _normalize(label: str) -> str:
    return " ".join(label.split()).casefold()


def convert_bpmn_to_pnml(bpmn_path: str, pnml_path: str) -> None:
    """Convert the BPMN model to a Petri net and write it as PNML."""
    bpmn = pm4py.read_bpmn(bpmn_path)
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn)
    pm4py.write_pnml(net, initial_marking, final_marking, pnml_path)


def check_constraints(
    ontology_path: str,
    bpmn_path: str,
    model_name: str,
    output_pnml: Optional[str] = None,
) -> ActivCheckResult:
    """
    Check the PartOf constraints of the process model `model_name` in the ontology at
    `ontology_path` against the BPMN model at `bpmn_path`.

    The BPMN is converted to a PNML file (`output_pnml`, default: next to the BPMN, named
    "<stem>.pnml"), which is loaded again to collect the labels of its transitions. For each
    PartOf constraint, the has-aggregate task must occur as a transition and none of the has-part
    tasks may. Tasks are matched to transitions on their rdfs:label (whitespace- and
    case-insensitive), falling back to the readable local name of their IRI.

    `model_name` is matched against the process model's rdfs:label (e.g. "Heart Failure"), falling
    back to the local name of its IRI; a KeyError is raised if the ontology has no such model.

    Returns an ActivCheckResult with one ActivConstraintResult per constraint. Like ltl_check.check_constraints,
    a failing constraint is data, not an exception; a constraint that is malformed in the ontology
    (not exactly one has-aggregate task) is reported as failing, with the reason in its `error`.
    """
    g = Graph()
    g.parse(str(ontology_path), format="turtle")

    if output_pnml is None:
        output_pnml = str(Path(bpmn_path).with_name(Path(bpmn_path).stem + ".pnml"))

    convert_bpmn_to_pnml(bpmn_path, output_pnml)
    net, _initial_marking, _final_marking = pm4py.read_pnml(output_pnml)

    # silent (tau) transitions have no label and are not activities
    activities = sorted({t.label for t in net.transitions if t.label is not None})
    occurring = {_normalize(a) for a in activities}

    candidates = list(g.subjects(RDFS.label, Literal(model_name))) + [CNSTR[model_name.replace(" ", "_")]]
    model = next((m for m in candidates if (m, RDF.type, CNSTR.ProcessModel) in g), None)
    if model is None:
        raise KeyError(f"No process model named {model_name!r} in the ontology")
    constraints = {c for c in g.objects(model, CNSTR.hasConstraint) if (c, RDF.type, CNSTR.PartOf) in g}

    results: List[ActivConstraintResult] = []

    for constraint in sorted(constraints, key=_natural_key):
        name = _label(constraint, g)
        aggregates = list(g.objects(constraint, CNSTR.hasAggregate))
        parts = sorted(_label(p, g) for p in g.objects(constraint, CNSTR.hasPart))

        if len(aggregates) != 1:
            results.append(ActivConstraintResult(
                name=name, holds=False, aggregate=", ".join(_label(a, g) for a in aggregates), parts=parts,
                aggregate_found=False,
                error=f"expected exactly one has-aggregate task, found {len(aggregates)}"))
            continue

        aggregate = _label(aggregates[0], g)
        aggregate_found = _normalize(aggregate) in occurring
        parts_found = [p for p in parts if _normalize(p) in occurring]

        results.append(ActivConstraintResult(
            name=name,
            holds=aggregate_found and not parts_found,
            aggregate=aggregate,
            parts=parts,
            aggregate_found=aggregate_found,
            parts_found=parts_found,
        ))

    return ActivCheckResult(pnml_path=output_pnml, activities=activities, results=results)
