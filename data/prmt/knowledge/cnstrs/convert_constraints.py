"""Query the constraints ontology (constraints.owl) for the LTL constraints of a process model."""
import argparse
import json
import re
import sys
import warnings
from enum import Enum
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef

CNSTR = Namespace("http://notation3.org/constraints.owl#")
LTL = Namespace("http://notation3.org/ltl_constraints.owl#")

PREFIXES = f"""
PREFIX rdf:  <{RDF}>
PREFIX rdfs: <{RDFS}>
PREFIX :     <{CNSTR}>
PREFIX ltl:  <{LTL}>
"""


class Syntax(Enum):
    """Output syntaxes. LOLA is the name of the annotation property (in the constraints namespace) holding
    the operator symbols of LoLA, and applies the rewrites LoLA needs (see ltl_formula_to_syntax). DEFAULT writes
    the formulas as they are in the ontology, with the same operator symbols but none of the LoLA-specific
    rewrites."""
    DEFAULT = "default"
    LOLA = "hasLolaSyntax"

    @property
    def symbol_property(self):
        """The annotation property holding the operator symbols of this syntax."""
        return CNSTR[Syntax.LOLA.value if self is Syntax.DEFAULT else self.value]


class Target(Enum):
    """What convert_constraints turns the constraints into."""
    SYNTAX = "syntax"
    NARRATION = "narration"


class OutputFormat(Enum):
    """How convert_constraints returns its result."""
    JSON_LIST = "json_list"
    STRING = "string"


@lru_cache(maxsize=None)
def load_ontology(path) -> Graph:
    g = Graph()
    g.parse(str(path), format="turtle")
    return g


def _natural_key(iri):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(iri))]


def create_ltl_constraints(ontology_path, constraints=None, check=True, overwrite=False):
    """Convert the instances of the Constraint subclasses (DecisionalDependency, Conflict, EntryPoint, ExitPoint,
    PrioritizedConditional, Alternative; PartOf has no LTL counterpart) into LtlFormula instances. An
    LtlConstraint already is one (its attached formulas are its content), so it is left as it is.

    The formulas are built from each constraint's own properties, following the shape of the LtlFormulas the
    ontology attaches to them (hasLtlFormula): e.g. a DecisionalDependency on tasks A, B becomes
    `NOT B U A` (decision not optional) or `G (F A -> (NOT B U A))` (decision optional).

    The ontology is loaded from `ontology_path`; the changes are made in a copy of it (the new graph). For each
    constraint, the formulas attached so far are removed (the hasLtlFormula links, and the formula nodes
    themselves, with anything else stated about them) and the created ones are attached instead, as individuals
    named `<constraint>_L<i>[_<path>]`.

    `constraints` limits the conversion to those constraint IRIs (default: every instance of a Constraint
    subclass other than LtlConstraint). With `check`, a warning is issued for every constraint whose created
    formulas differ from the ones it replaces (compared structurally), so inconsistencies between a
    constraint's properties and its attached formulas show up.

    With `overwrite`, the old ontology is overwritten with the new graph, i.e. the new graph is written (as
    Turtle) to `ontology_path`. Note that this re-serializes the file, so its formatting and comments are not
    preserved. Without `overwrite`, the file is left untouched.

    Returns the updated graph.
    """
    old = load_ontology(Path(ontology_path))
    g = Graph()
    for prefix, namespace in old.namespaces():
        g.bind(prefix, namespace)
    g += old
    if constraints is None:
        constraints = {c for cls in g.transitive_subjects(RDFS.subClassOf, CNSTR.Constraint)
                       if cls != CNSTR.Constraint
                       for c in g.subjects(RDF.type, cls)}
    for constraint in sorted(constraints, key=_natural_key):
        constraint = URIRef(constraint)
        if (constraint, RDF.type, CNSTR.LtlConstraint) in g:
            continue
        specs = _constraint_specs(constraint, g)
        previous = sorted(g.objects(constraint, CNSTR.hasLtlFormula), key=_natural_key)
        if check:
            old_trees = sorted(repr(_formula_tree(f, g)) for f in previous)
            old_syntax = [ltl_formula_to_syntax(f, g) for f in previous]

        for formula in previous:
            _remove_formula(g, formula)
        g.remove((constraint, CNSTR.hasLtlFormula, None))

        formulas = []
        for i, spec in enumerate(specs, 1):
            root = URIRef(f"{constraint}_L{i}")
            if (root, None, None) in g:
                raise ValueError(f"cannot attach a new formula to {_label(constraint, g)}: {root} already exists")
            formulas.append(_add_formula(g, root, spec))
            g.add((constraint, CNSTR.hasLtlFormula, root))

        if check and old_trees != sorted(repr(_formula_tree(f, g)) for f in formulas):
            warnings.warn(f"{_label(constraint, g)}: the formulas created from its properties differ from the "
                          f"ones it had attached: created {[ltl_formula_to_syntax(f, g) for f in formulas]}, "
                          f"previously attached {old_syntax}", stacklevel=2)

    if overwrite:
        g.serialize(destination=str(ontology_path), format="turtle")
        load_ontology.cache_clear()  # the cached graph of this file is stale now
    return g


def _constraint_specs(c, g):
    """The formulas (as nested tuples of operator names, ending in Task/atom IRIs) of constraint `c`."""
    types = set(g.objects(c, RDF.type))

    def obj(prop):
        value = g.value(c, prop)
        if value is None:
            raise ValueError(f"{_label(c, g)} has no {str(prop).rsplit('#', 1)[-1]}")
        return value

    def before(a, b, optional):
        """B only after A: `NOT B U A`, or, if A is optional, `G (F A -> (NOT B U A))`."""
        until = ("Until", ("Negation", b), a)
        return ("Globally", ("Implies", ("Eventually", a), until)) if optional else until

    if CNSTR.DecisionalDependency in types:
        optional = bool(g.value(c, CNSTR.isDecisionOptional))
        return [before(obj(CNSTR.hasDecisionTask), obj(CNSTR.hasDependentTask), optional)]
    if CNSTR.Conflict in types:
        # Conflict has no isDecisionOptional in the ontology, yet its attached formulas use the optional form
        return [before(obj(CNSTR.hasInterferedTask), obj(CNSTR.hasInterferingTask), True)]
    if CNSTR.EntryPoint in types:
        return [("Next", obj(CNSTR.hasTask))]
    if CNSTR.ExitPoint in types:
        return [("Eventually", ("And", obj(CNSTR.hasTask), ("Next", CNSTR["last"])))]
    if CNSTR.PrioritizedConditional in types:
        q, z, mark = obj(CNSTR.hasDecisionTask), obj(CNSTR.hasConditionalTask), obj(CNSTR.hasConditionMarker)
        return [("Globally", ("Implies", mark, ("Next", z))),  # marker -> conditional task right after
                ("Globally", ("Implies", ("Next", mark), q))]  # marker only right after the decision task
    if CNSTR.Alternative in types:
        replacement = obj(CNSTR.hasReplacement)
        return [("Implies", ("Eventually", replaced), ("Negation", ("Eventually", replacement)))
                for replaced in sorted(g.objects(c, CNSTR.hasReplaced), key=_natural_key)]
    if CNSTR.PartOf in types:
        return []
    raise ValueError(f"{c} is not an instance of a known Constraint subclass")


def _remove_formula(g, formula):
    """Remove the operator nodes of the formula (IRI) from the graph, with every triple about them; the tasks
    and atoms it refers to are kept."""
    if not any(LTL.LtlOperator in g.transitive_objects(t, RDFS.subClassOf) for t in g.objects(formula, RDF.type)):
        return
    operands = [o for p in (LTL.hasOperand, LTL.hasLeftOperand, LTL.hasRightOperand) for o in g.objects(formula, p)]
    g.remove((formula, None, None))
    g.remove((None, None, formula))
    for operand in operands:
        _remove_formula(g, operand)


def _add_formula(g, iri, spec):
    """Add the formula `spec` to the graph under `iri`; returns its IRI. Nested operators are named `<iri>_<n>`."""
    if isinstance(spec, URIRef):
        return spec
    node = URIRef(iri)
    kind, operands = spec[0], spec[1:]
    g.add((node, RDF.type, LTL[kind]))
    properties = (LTL.hasOperand,) if len(operands) == 1 else (LTL.hasLeftOperand, LTL.hasRightOperand)
    for i, (prop, operand) in enumerate(zip(properties, operands), 1):
        g.add((node, prop, _add_formula(g, f"{iri}_{i}", operand)))
    return node


def convert_constraint(constraint, graph, target: Target = Target.SYNTAX, out_syntax: Syntax = Syntax.DEFAULT):
    """Return the given constraint (IRI; an instance of a Constraint subclass) as a list of JSON objects,
    structured like the constraint files in this folder.

    For Target.SYNTAX these are its attached LtlFormulas (hasLtlFormula), written in `out_syntax`; for an
    LtlConstraint, these formulas are all there is to the constraint:
    [{"name": ..., "type": "raw", "formula": ...}, ...]. If the constraint has several formulas, their names
    get a numeric suffix. Constraints without a formula (e.g. PartOf) give an empty list; they can only be
    narrated (Target.NARRATION).

    For Target.NARRATION the list holds the constraint's narration (`out_syntax` is then unused; for an
    LtlConstraint, the narrations of its formulas, one per line):
    [{"name": ..., "type": "narration", "narration": ...}].

    `name` is the constraint's rdfs:label (the original name, e.g. "cap1"), falling back to the
    readable local name of its IRI (e.g. "CAP 1").
    """
    g = graph
    constraint = URIRef(constraint)
    target = Target(target)
    name = _label(constraint, g)

    formulas = sorted(g.objects(constraint, CNSTR.hasLtlFormula), key=_natural_key)

    if target is Target.NARRATION:
        if (constraint, RDF.type, CNSTR.LtlConstraint) in g:
            narration = "\n".join(ltl_formula_to_narration(f, g) for f in formulas)
        else:
            narration = ontology_constraint_to_narration(constraint, g)
        return [{"name": name, "type": "narration", "narration": narration}]

    names = [name] if len(formulas) == 1 else [f"{name}_{i}" for i in range(1, len(formulas) + 1)]
    return [{"name": n, "type": "raw", "formula": _toplevel_formula_to_syntax(formula, g, out_syntax)}
            for n, formula in zip(names, formulas)]


def convert_constraints(model_name, graph, target: Target = Target.SYNTAX, out_syntax: Syntax = Syntax.DEFAULT,
                        out_path=None, out_format: OutputFormat = OutputFormat.JSON_LIST):
    """Return the constraints of the named process model as a JSON array, structured like
    the constraint files in this folder: [{"name": ..., "type": "raw", "formula": ...}, ...].

    Each constraint (an instance of a Constraint subclass) is converted by
    convert_constraint. For Target.SYNTAX these are the LtlFormulas (hasLtlFormula) of constraints such as
    DecisionalDependency; if a constraint has several formulas, their names get a numeric suffix.
    Constraints without a formula (e.g. PartOf) can only be narrated (Target.NARRATION).

    `target` is Target.SYNTAX (or "syntax") to write each constraint as a formula in `out_syntax`, or
    Target.NARRATION (or "narration") to write it as a human-readable narration (`out_syntax` is then
    unused); narrations are given as [{"name": ..., "type": "narration", "narration": ...}, ...].

    `name` is the constraint's rdfs:label (the original name, e.g. "cap1"), falling back to the
    readable local name of its IRI (e.g. "CAP 1").
    `out_syntax` selects how the operators are written (Syntax.DEFAULT: without any LoLA-specific rewrites).
    `out_format` is OutputFormat.JSON_LIST (the JSON array described above) or OutputFormat.STRING (a
    single string with the constraints, i.e. the formulas or the narrations, separated by a blank line).
    Constraints with identical output (formula or narration) are only included once, the first one kept.
    If `out_path` is given, the result is also written to that path (as JSON, or as that string).

    `model_name` is matched against the process model's rdfs:label (e.g. "Stroke 1"),
    falling back to the local name of its IRI (e.g. "Stroke_1").
    """
    g = graph
    target = Target(target)
    out_format = OutputFormat(out_format)

    models = list(g.query(
        PREFIXES + "SELECT DISTINCT ?m WHERE { ?m a :ProcessModel ; rdfs:label ?name }",
        initBindings={"name": Literal(model_name)},
    ))
    if models:
        model = models[0][0]
    elif (CNSTR[model_name], RDF.type, CNSTR.ProcessModel) in g:
        model = CNSTR[model_name]
    else:
        raise KeyError(f"No process model named {model_name!r} in the ontology")

    rows = g.query(
        PREFIXES + """
        SELECT DISTINCT ?c WHERE {
            ?m :hasConstraint ?c .
            ?c a/rdfs:subClassOf* :Constraint .
        }""",
        initBindings={"m": model},
    )
    constraints = sorted((r[0] for r in rows), key=_natural_key)
    result = [entry for c in constraints for entry in convert_constraint(c, g, target, out_syntax)]

    # drop constraints whose formula/narration repeats an earlier one (the first is kept)
    text_key = "formula" if target is Target.SYNTAX else "narration"
    seen = set()
    result = [c for c in result if not (c[text_key] in seen or seen.add(c[text_key]))]

    if out_format is OutputFormat.STRING:
        result = "\n\n".join(c[text_key] for c in result)

    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            if out_format is OutputFormat.STRING:
                f.write(result)
            else:
                json.dump(result, f, indent=4, ensure_ascii=False)
    return result


def convert_all_constraints(ontology_path, target: Target = Target.SYNTAX,
                            out_syntax: Syntax = Syntax.DEFAULT, out_path=None,
                            out_format: OutputFormat = OutputFormat.JSON_LIST):
    """Run convert_constraints for every process model in the ontology file at `ontology_path`.

    `target`, `out_syntax` and `out_format` are passed on to convert_constraints.
    Returns {process model name: convert_constraints result}.
    If `out_path` is given, it is a directory (created if needed) that receives one file per process
    model, named after the model (`<model>.json` for JSON_LIST, `<model>.txt` for STRING).
    """
    g = load_ontology(Path(ontology_path))
    out_format = OutputFormat(out_format)

    names = sorted(str(g.value(m, RDFS.label) or str(m).rsplit("#", 1)[-1])
                   for m in g.subjects(RDF.type, CNSTR.ProcessModel))
    if out_path is not None:
        Path(out_path).mkdir(parents=True, exist_ok=True)
    extension = "txt" if out_format is OutputFormat.STRING else "json"

    return {
        name: convert_constraints(
            name, g, target, out_syntax, Path(out_path) / f"{name}.{extension}" if out_path is not None else None,
            out_format)
        for name in names
    }


def ltl_formula_to_syntax(formula, graph, out_syntax: Syntax = Syntax.DEFAULT):
    """Convert the LtlFormula (IRI) into its string representation, following the ontology's properties.

    Operators are written using the annotation property of `out_syntax` (e.g. Syntax.LOLA -> `hasLolaSyntax`; Syntax.DEFAULT uses the same symbols, but applies none of LoLA's
    rewrites: no X/XX handling, no `@end`, no mutex or G rewrites); binary operators are always
    parenthesised, e.g. `G (F "A" -> (NOT "B" U "A"))`.
    """
    g = graph
    formula = URIRef(formula)
    syntax_property = out_syntax.symbol_property

    types = set(g.objects(formula, RDF.type))

    if out_syntax is Syntax.LOLA:
        rewritten = _lola_mutex_rewrite(formula, types, g)
        if rewritten is not None:
            return rewritten

    if CNSTR.Task in types:
        return f'"{_label(formula, g)}"'
    if LTL.LtlAtom in types:
        if out_syntax is Syntax.LOLA and formula == CNSTR["last"]:
            return "@end"
        return _label(formula, g)

    symbol = next((str(s) for t in types for s in g.objects(t, syntax_property)), None)
    if symbol is None:
        raise ValueError(f"{formula} is not a known LTL operator, task, or atom")
    if out_syntax is Syntax.LOLA and symbol == "X":
        # an outer Next (i.e. a constraint of a process model, or the formula of a constraint that is)
        # over a single task is one step; otherwise it is two
        is_outer = (None, CNSTR.hasConstraint, formula) in g or any(
            (None, CNSTR.hasConstraint, c) in g for c in g.subjects(CNSTR.hasLtlFormula, formula))
        symbol = "X" if is_outer and _count_tasks(formula, g) == 1 else "XX"

    operand = g.value(formula, LTL.hasOperand)
    if operand is not None:
        return f"{symbol} {ltl_formula_to_syntax(operand, g, out_syntax)}"

    left, right = g.value(formula, LTL.hasLeftOperand), g.value(formula, LTL.hasRightOperand)
    if left is None or right is None:
        raise ValueError(f"{formula} is missing an operand")
    return f"({ltl_formula_to_syntax(left, g, out_syntax)} {symbol} {ltl_formula_to_syntax(right, g, out_syntax)})"


_TEMPORAL_KINDS = {"Globally", "Eventually", "Next", "Until", "WeakUntil", "Release"}


def _toplevel_formula_to_syntax(formula, graph, out_syntax: Syntax = Syntax.DEFAULT):
    """ltl_formula_to_syntax for the outermost formula of a constraint, which -- for LoLA -- needs care that
    nested formulas don't: LoLA splits a top-level `->` or `OR` between two temporal formulas into two
    separate "holds on all runs" checks (each side gets its own Buechi automaton) instead of checking every
    run against the whole formula. E.g. `F "A" -> (NOT "B" U "A")` is then read as "on every run "A" never
    happens, or on every run NOT "B" U "A"" -- which fails on a model where only some runs contain "A" (the
    strong U demands "A" on the runs that don't). Inside a G, or as a single temporal formula, the same
    connective is checked per run, as intended.

    So a known shape is wrapped in G (see _lola_split_rewrite), and any other such top-level `->`/`OR` gets a
    warning.
    """
    if out_syntax is Syntax.LOLA:
        formula = URIRef(formula)
        rewritten = _lola_split_rewrite(formula, graph)
        if rewritten is not None:
            return rewritten

        types = set(graph.objects(formula, RDF.type))
        if _lola_mutex_rewrite(formula, types, graph) is None:
            tree = _formula_tree(formula, graph)
            if tree[0] in ("Implies", "Or") and all(_has_temporal(t) for t in tree[1:]):
                warnings.warn(
                    f"{_label(formula, graph)}: LoLA checks a top-level {tree[0]} between temporal formulas as "
                    "two separate 'holds on all runs' properties, which is not what the formula means; rewrite "
                    "it as a single temporal formula or wrap it in G", stacklevel=2)

    return ltl_formula_to_syntax(formula, graph, out_syntax)


def _has_temporal(tree):
    return tree[0] in _TEMPORAL_KINDS or any(isinstance(t, tuple) and _has_temporal(t) for t in tree[1:])


def _lola_split_rewrite(formula, g):
    """For a formula of the shape `F "A" -> (NOT B U "A")` ("if A ever happens, then not B until A"), return the
    LoLA syntax of the same formula wrapped in a G, `G (F "A" -> (NOT B U "A"))`, or None if `formula` doesn't
    match that shape. This is the form the ontology already uses for such constraints, and LoLA checks the
    implication per run inside the G.
    """
    if LTL.Implies not in set(g.objects(formula, RDF.type)):
        return None

    left, right = g.value(formula, LTL.hasLeftOperand), g.value(formula, LTL.hasRightOperand)
    if left is None or right is None:
        return None
    if LTL.Eventually not in set(g.objects(left, RDF.type)) or LTL.Until not in set(g.objects(right, RDF.type)):
        return None

    a = g.value(left, LTL.hasOperand)
    negation, until_a = g.value(right, LTL.hasLeftOperand), g.value(right, LTL.hasRightOperand)
    if a is None or negation is None or until_a is None:
        return None
    if LTL.Negation not in set(g.objects(negation, RDF.type)) or g.value(negation, LTL.hasOperand) is None:
        return None
    if _formula_tree(a, g) != _formula_tree(until_a, g):  # the same A on both sides
        return None

    return f"G {ltl_formula_to_syntax(formula, g, Syntax.LOLA)}"


def _lola_mutex_rewrite(formula, types, g):
    """For a formula of the shape `F X -> NOT F Y` (mutual exclusion between X and Y over the
    whole trace), return the LoLA syntax of the equivalent `G (X -> NOT F Y) AND G (Y -> NOT F X)`
    instead, or None if `formula` doesn't match that shape.

    LoLA's LTL/Buechi engine has been observed to answer this raw shape incorrectly -- e.g.
    reporting a violation whose witness path never actually fires X -- on nets using this
    ontology's event_per_state encoding (transient per-firing event places), even though X and Y
    are structurally alternative (XOR-gateway) tasks that can never both occur. The G-wrapped
    reformulation is logically equivalent (checked exhaustively over the FX/FY truth-table) but
    is model-checked correctly, matching the shape of the response/precedence formulas that
    already work reliably.
    """
    if LTL.Implies not in types:
        return None

    left, right = g.value(formula, LTL.hasLeftOperand), g.value(formula, LTL.hasRightOperand)
    if left is None or right is None:
        return None

    if LTL.Eventually not in set(g.objects(left, RDF.type)):
        return None
    if LTL.Negation not in set(g.objects(right, RDF.type)):
        return None

    not_operand = g.value(right, LTL.hasOperand)
    if not_operand is None or LTL.Eventually not in set(g.objects(not_operand, RDF.type)):
        return None

    x, y = g.value(left, LTL.hasOperand), g.value(not_operand, LTL.hasOperand)
    if x is None or y is None:
        return None

    x_s, y_s = ltl_formula_to_syntax(x, g, Syntax.LOLA), ltl_formula_to_syntax(y, g, Syntax.LOLA)
    return f"(G ({x_s} -> NOT F {y_s}) AND G ({y_s} -> NOT F {x_s}))"


def ontology_constraint_to_narration(constraint, graph):
    """Convert the constraint (IRI) into a human-readable narration, based on its ontology class and
    properties (one dedicated template per class: DecisionalDependency, Conflict, EntryPoint, ExitPoint,
    PrioritizedConditional, PartOf and Alternative). Returns None if the constraint's class has none of
    these (e.g. an LtlConstraint; see ltl_formula_to_narration for its formulas).
    """
    g = graph
    constraint = URIRef(constraint)
    types = set(g.objects(constraint, RDF.type))

    def task(prop):
        return f'"{_label(g.value(constraint, prop), g)}"'

    # "B" may only occur after "A" takes place / potentially takes place
    if CNSTR.DecisionalDependency in types:
        a, b = task(CNSTR.hasDecisionTask), task(CNSTR.hasDependentTask)
        optional = bool(g.value(constraint, CNSTR.isDecisionOptional))
        return f'{b} may only occur after {a} {"potentially " if optional else ""}takes place.'

    # "B" may only occur after "A" potentially takes place (a Conflict has no isDecisionOptional: the interfered
    # task is always optional, as in the formulas attached to it)
    if CNSTR.Conflict in types:
        a, b = task(CNSTR.hasInterferedTask), task(CNSTR.hasInterferingTask)
        return f'{b} may only occur after {a} potentially takes place.'

    # "A" must be the first task in the process
    if CNSTR.EntryPoint in types:
        return f'{task(CNSTR.hasTask)} must be the first task in the process.'

    # "A" should be the last task in the process
    if CNSTR.ExitPoint in types:
        return f'{task(CNSTR.hasTask)} should be the last task in the process.'

    # right after "Q", in case the condition "Y" is met, insert an extra activity "A"; then "Z" must follow "A"
    if CNSTR.PrioritizedConditional in types:
        q, y, z, a = task(CNSTR.hasDecisionTask), task(CNSTR.hasCondition), task(CNSTR.hasConditionalTask), \
            task(CNSTR.hasConditionMarker)
        return (f'Right after {q}, in case the condition {y} is met, insert an extra activity {a}.\n'
                f'If {a} occurs, then {z} must take place immediately afterwards. This is an absolute priority.')

    # "A" is part of "Y" (one line for each part of the aggregate)
    if CNSTR.PartOf in types:
        y = task(CNSTR.hasAggregate)
        parts = sorted(_label(part, g) for part in g.objects(constraint, CNSTR.hasPart))
        return "\n".join(f'"{part}" is part of {y}.' for part in parts)

    # "Y" is an alternative for any of the tasks "A", "B", ..., "K"
    if CNSTR.Alternative in types:
        y = task(CNSTR.hasReplacement)
        replaced = sorted(f'"{_label(r, g)}"' for r in g.objects(constraint, CNSTR.hasReplaced))
        return f'{y} is an alternative for any of the tasks {", ".join(replaced)}.'

    return None


def ltl_formula_to_narration(formula, graph):
    """Convert the LtlFormula (IRI) into a human-readable narration, based on the ontology's contents.

    Formulas matching one of the known constraint patterns get a dedicated template; any other formula
    gets a generic, operator-by-operator paraphrase. (Constraints whose class ontology_constraint_to_
    narration recognizes, e.g. DecisionalDependency or PartOf, are narrated by that function instead.)
    """
    g = graph
    tree = _formula_tree(URIRef(formula), g)

    # "B" may only occur after "A" takes place
    m = _not_until(tree)
    if m:
        b, a = m
        return f'{b} may only occur after {a} takes place.'

    # "B" may only occur after "A" potentially takes place
    if tree[0] == "Globally" and tree[1][0] == "Implies" and tree[1][1][0] == "Eventually":
        m = _not_until(tree[1][2])
        if m and tree[1][2][2] == tree[1][1][1]:  # the same task "A" on both sides
            b, a = m
            return f'{b} may only occur after {a} potentially takes place.'

    # "A" must be the first task in the process
    if tree[0] == "Next" and tree[1][0] == "Task":
        a = _phrase(tree[1])
        return f'{a} must be the first task in the process.'

    # "A" should be the last task in the process
    if (tree[0] == "Eventually" and tree[1][0] == "And" and tree[1][1][0] == "Task"
            and tree[1][2] == ("Next", ("Atom", "last"))):
        a = _phrase(tree[1][1])
        return f'{a} should be the last task in the process.'

    text = _phrase(tree)
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return text[0].upper() + text[1:] + "."


def _formula_tree(formula, g):
    """The formula as nested tuples: ("Task", label), ("Atom", label), (operator, operand) or (operator, left, right)."""
    types = set(g.objects(formula, RDF.type))
    if CNSTR.Task in types:
        return ("Task", _label(formula, g))
    if LTL.LtlAtom in types:
        return ("Atom", _label(formula, g))

    kind = next((str(t).rsplit("#", 1)[-1] for t in types
                 if (t, RDFS.subClassOf, LTL.UnaryOperator) in g or (t, RDFS.subClassOf, LTL.BinaryOperator) in g), None)
    if kind is None:
        raise ValueError(f"{formula} is not a known LTL operator, task, or atom")

    operand = g.value(formula, LTL.hasOperand)
    if operand is not None:
        return (kind, _formula_tree(operand, g))
    left, right = g.value(formula, LTL.hasLeftOperand), g.value(formula, LTL.hasRightOperand)
    if left is None or right is None:
        raise ValueError(f"{formula} is missing an operand")
    return (kind, _formula_tree(left, g), _formula_tree(right, g))


def _not_until(tree):
    """For NOT "B" U "A" (tasks only): returns the quoted (B, A), else None."""
    if tree[0] == "Until" and tree[1][0] == "Negation" and tree[1][1][0] == "Task" and tree[2][0] == "Task":
        return _phrase(tree[1][1]), _phrase(tree[2])
    return None


# generic paraphrase per node kind; "last" is the special atom marking the end of the process
_PHRASES = {
    "Task": '"{}"', "Atom": "{}", "last": "the end of the process",
    "Negation": "not {}", "Globally": "at all times, {}", "Eventually": "eventually, {}", "Next": "in the next step, {}",
    "And": "({} and {})", "Or": "({} or {})", "Implies": "(if {}, then {})", "Until": "({} until {})",
    "WeakUntil": "({} weakly until {})", "Release": "({} releases {})",
}


def _phrase(tree):
    """Generic paraphrase of a formula tree."""
    kind = "last" if tree == ("Atom", "last") else tree[0]
    return _PHRASES[kind].format(*(_phrase(t) if isinstance(t, tuple) else t for t in tree[1:]))


def _count_tasks(formula, g):
    """Number of Task occurrences in the formula (the formula itself included)."""
    if CNSTR.Task in set(g.objects(formula, RDF.type)):
        return 1
    return sum(_count_tasks(o, g) for p in (LTL.hasOperand, LTL.hasLeftOperand, LTL.hasRightOperand)
               for o in g.objects(formula, p))


def _label(iri, g):
    label = g.value(iri, RDFS.label)
    return str(label) if label is not None else unquote(str(iri).rsplit("#", 1)[-1]).replace("_", " ")


def _enum_type(enum):
    """argparse type: a (case-insensitive) member name of `enum`."""
    def parse(text):
        try:
            return enum[text.upper()]
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"invalid choice: {text!r} (choose from {', '.join(m.name.lower() for m in enum)})") from None
    return parse


# convert_cnstr.py ontology_file [-m MODEL] [-t {syntax,narration}] [-s {default,lola}] [-f {json_list,string}] [-o OUT_PATH]
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert the LTL constraints of process models in a constraints ontology. "
                    "Without --model, all process models in the ontology are converted.")
    parser.add_argument("ontology", help="path of the ontology file")
    parser.add_argument("-m", "--model", help="name of the process model to convert (default: all process models)")
    parser.add_argument("-t", "--target", type=_enum_type(Target), default=Target.SYNTAX,
                        metavar="{" + ",".join(m.name.lower() for m in Target) + "}",
                        help="convert to formulas or to narrations (default: syntax)")
    parser.add_argument("-s", "--out-syntax", type=_enum_type(Syntax), default=Syntax.DEFAULT,
                        metavar="{" + ",".join(m.name.lower() for m in Syntax) + "}",
                        help="syntax of the formulas, for --target syntax (default: default)")
    parser.add_argument("-f", "--out-format", type=_enum_type(OutputFormat), default=OutputFormat.JSON_LIST,
                        metavar="{" + ",".join(m.name.lower() for m in OutputFormat) + "}",
                        help="JSON array, or one string with blank-line-separated constraints (default: json_list)")
    parser.add_argument("-o", "--out-path",
                        help="save the result to this path; with all process models, a directory receiving one file "
                             "per model (the result is then not printed)")
    args = parser.parse_args(argv)

    graph = load_ontology(args.ontology)
    options = dict(target=args.target, out_syntax=args.out_syntax, out_path=args.out_path,
                   out_format=args.out_format)

    if args.model is not None:
        result = convert_constraints(args.model, graph, **options)
        results = {args.model: result}
    else:
        results = convert_all_constraints(args.ontology, **options)
        result = results

    if args.out_path is not None:
        print(f"Saved to {args.out_path}")
    elif args.out_format is OutputFormat.STRING:
        print("\n\n".join(r if args.model is not None else f"== {name}\n\n{r}" for name, r in results.items()))
    else:
        print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
