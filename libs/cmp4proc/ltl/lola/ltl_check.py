"""
ltl_check.py (LoLA backend)

End-to-end BPMN/PNML + LTL validation using the LoLA 2.0 model checker:

    BPMN or PNML
      |
    PM4Py Petri net --------> SVG visualization (pm4py)
      |               `-----> PNML export (pm4py)
      |
    LoLA net (+ event places, see below)
      |
    LoLA model checking
      |
    TRUE / FALSE (+ witness path, when LoLA produces one)

LoLA model-checks CTL/LTL formulas directly over a Petri net's *markings* --
it has no built-in notion of "the transition that just fired" as an
atomic proposition (only FIREABLE(t), which means "t is currently
enabled", not "t has fired"). To express Declare-style constraints such
as response/precedence/existence/absence in terms of activities, this
module augments the net with places that expose "activity X occurred"
to formulas -- see construct_model()'s `event_semantics` for the full
explanation, but in short, there are two selectable encodings:

- "event_per_state" (the default): each visible transition is split
  into an emit step (produces only a pulse place `evt_<t>`, true for
  exactly one state) and a reset step (consumes the pulse, produces
  what the transition actually used to produce). This gives real,
  non-monotonic event-trace semantics -- `response`/`existence`/
  `absence` are exact regardless of how many times an activity recurs.
  `precedence`/`succession` compile to a different formula under this
  mode (see compile_constraint()), and any hand-written "raw" formula
  using X (nextstate) needs to account for the extra reset step -- see
  construct_model()'s docstring for a worked example.

- "sticky_events": one extra place per visible transition (`occ_<t>`)
  that the transition produces into every time it fires, on top of
  whatever it already produces -- a strictly additive change (an extra
  PRODUCE arc into a fresh, unbounded place never blocks a transition
  from firing) rather than a restructuring. `occ_<t>` is monotonic --
  once true, always true -- so `response`/`precedence`/`succession`
  here mean "some occurrence of A/B satisfies the relation" rather
  than "every occurrence does". For workflow nets where each activity
  fires at most once per process instance (the common case for acyclic
  BPMN diagrams) this is exact; for a model with loops that let an
  activity recur, it's a sound but weaker approximation.

Requirements
------------
pip install -U pm4py

LoLA must be built; see src/ltl/lola/lola-2.0/src/lola (already built in
this checkout) or https://theo.informatik.uni-rostock.de/theo-forschung/tools/lola/

Example
-------
python ltl_check.py check model.bpmn
python ltl_check.py check model.pnml --lola /path/to/lola

Constraints normally come from the CONSTRAINTS list below, but can
instead be loaded from a JSON/YAML file (same schema, a list of
constraint dicts) with --constraints:

python ltl_check.py model.bpmn --constraints constraints.yaml

Library usage
-------------
Client code should call check_constraints() directly rather than going
through the CLI/subprocess -- it returns a LolaCheckResult (generated model
+ per-constraint PASS/FAIL/witness) as data, and never calls sys.exit():

from ltl_check import check_constraints

result = check_constraints("model.bpmn", CONSTRAINTS, lola_command="lola")
for r in result.results:
    print(r.name, r.holds)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pm4py
from pm4py.objects.petri_net.obj import Marking, PetriNet

from libs.cmp4proc.check_result import CheckResult, ConstraintResult
from libs.cmp4proc.ltl.lola.pnml_to_lola import LolaNames, assign_names, render_lola


# ============================================================
# Configuration
# ============================================================

# Define your constraints here.
#
# These use the ORIGINAL BPMN/PNML activity labels, not LoLA identifiers.
#
# Supported (exact under the default event_semantics="event_per_state";
# see the module docstring for the "at least one occurrence" caveat that
# applies instead under event_semantics="sticky_events"):
#
#   response:
#       Whenever A occurs, B eventually occurs.
#       G(A -> F B)
#
#   precedence:
#       B may occur only if A occurred previously.
#       (compiles differently per event_semantics -- see
#       compile_constraint()/_precedence_formula())
#
#   existence:
#       A must occur at least once.
#       F A
#
#   absence:
#       A must never occur.
#       G !A
#
#   succession:
#       combines response(A,B) and precedence(A,B)
#
#   raw:
#       Advanced escape hatch: an arbitrary LoLA CTL/LTL formula string,
#       written directly in terms of place names / FIREABLE(...) / etc.
#       as they appear in the generated .lola file. "Double-quoted
#       activity labels" are expanded to an event-place check, e.g.:
#           G ("Take blood culture" -> F "Administer antibiotics")
#       and `@end` to "the process has reached its final marking", e.g.:
#           F ("Discharge patient" AND XX @end)
#       (see substitute_activities()).

CONSTRAINTS = [
    # Example:
    #
    # {
    #     "name": "sepsis_requires_antibiotics",
    #     "type": "response",
    #     "a": "Identify sepsis",
    #     "b": "Administer antibiotics",
    # },

    # Example:
    #
    # {
    #     "name": "culture_before_antibiotics",
    #     "type": "precedence",
    #     "a": "Take blood culture",
    #     "b": "Administer antibiotics",
    # },

]


# ============================================================
# Data structures
# ============================================================

@dataclass
class LolaModel:
    net_text: str
    names: LolaNames
    activity_symbols: Dict[str, str]  # BPMN/PNML activity label -> occ_<t> place id
    # LoLA condition that is true exactly once the process has reached its
    # final marking, e.g. "(p27_sink >= 1)"; what the `@end` placeholder in
    # raw formulas expands to. None if the net has no final marking.
    end_condition: Optional[str] = None


@dataclass
class LolaConstraintResult(ConstraintResult):
    formula: str
    witness: str  # LoLA's witness path, if it produced one for a failing check


@dataclass
class LolaCheckResult(CheckResult):
    model: LolaModel
    lola_path: str
    svg_path: str


# ============================================================
# 1. BPMN/PNML -> PM4Py Petri net -> LoLA net (+ event places)
# ============================================================

def load_petri_net(
    source_file: str,
    verbose: bool = True,
) -> Tuple[PetriNet, Marking, Marking]:

    suffix = Path(source_file).suffix.lower()

    if suffix == ".bpmn":

        if verbose:
            print("[1] Reading BPMN model")
        bpmn = pm4py.read_bpmn(source_file)

        if verbose:
            print("[2] Converting BPMN to Petri net")
        return pm4py.convert_to_petri_net(bpmn)

    elif suffix == ".pnml":

        if verbose:
            print("[1] Reading PNML Petri net")
        return pm4py.read_pnml(source_file, auto_guess_final_marking=True)

    else:
        raise ValueError(
            f"Unsupported input file type '{suffix}': expected .bpmn or .pnml"
        )


def visualize_petri_net(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    svg_path: str,
    names: LolaNames,
) -> None:
    """
    Render the converted Petri net to an SVG file via pm4py, labeling
    each place/transition with the same identifier it gets in the
    generated .lola file (rather than pm4py's own, often UUID-derived,
    internal name) -- so a LoLA formula like `occ_t0_B >= 1` can be
    matched back to a box in the diagram by eye.

    Note: places/transitions in the initial or final marking are always
    drawn by pm4py as plain start/end symbols with no text label at
    all, regardless of decorations -- that's a pm4py rendering rule,
    not something this function can override.
    """

    decorations = {
        place: {"label": names.places[place.name]} for place in net.places
    }

    decorations.update(
        {
            transition: {"label": names.transitions[transition.name]}
            for transition in net.transitions
        }
    )

    pm4py.save_vis_petri_net(
        net,
        initial_marking,
        final_marking,
        svg_path,
        decorations=decorations,
        debug=False,
    )


def save_petri_net_pnml(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    pnml_path: str,
) -> None:
    """Save the (intermediate, converted) Petri net as a PNML file via pm4py."""

    pm4py.write_pnml(net, initial_marking, final_marking, pnml_path)


def create_activity_symbols(names: LolaNames, prefix: str) -> Dict[str, str]:
    """
    Map visible (labeled) transitions' BPMN/PNML activity labels to the
    LoLA identifier of their event place: "evt_<t>" for the default
    event_per_state (pulse) semantics, "occ_<t>" for the sticky_events
    alternative -- see construct_model()'s `event_semantics`.
    """

    return {
        label: f"{prefix}_{transition_id}"
        for label, transition_id in names.activities.items()
    }


def construct_model(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    names: LolaNames,
    verbose: bool = True,
    event_semantics: str = "event_per_state",
) -> LolaModel:
    """
    Build the LoLA net, choosing how "activity X occurred" is exposed
    to formulas via `event_semantics`:

    "event_per_state" (default): a genuine "just fired" flag (`evt_<t>`)
    that is true for exactly one state, then false again -- real,
    non-monotonic event-trace semantics. Achieved by splitting each
    visible transition into an emit step (fires exactly as before, but
    now produces *only* its pulse place) and a reset step (consumes
    the pulse, produces what the transition actually used to produce).
    Downstream places only ever come from the reset, so nothing can
    race ahead of it before the pulse is observed. This makes
    response/existence/absence exact (every occurrence, not "at least
    one"); precedence needs a different formula under this semantics
    (see compile_constraint()), since it can no longer rely on
    stickiness to mean "already happened".

    "sticky_events": a monotonic per-activity counter (`occ_<t>`) that
    a transition produces into every time it fires, on top of whatever
    it already produces -- purely additive, never changes what the
    original net can do. Once set it stays set, so `occ_X >= 1` means
    "X occurred at least once so far", not "X just occurred". This is
    the semantics documented (with its "at least one occurrence"
    caveat) in this module's docstring.

    Gotcha for hand-written "raw" formulas using X (nextstate) under
    "event_per_state": the reset step means a pulse and the real
    effect it used to produce directly are now two states apart, not
    one. "Right after A comes B" is `X X "B"` (not `X "B"`) -- one X
    to settle through A's own reset, a second to reach B's pulse.
    Checking something about the *settled result* itself (e.g. "A
    directly produces the final marking", via FIREABLE(t_end_loop))
    only needs one X, since that result *is* what the reset produces.
    Note that the end of the run is a 2-cycle (t_end_loop and
    t_end_loop_back alternate), so FIREABLE(t_end_loop) alone is true in
    only every other end state. Use `@end` (see substitute_activities())
    for "the process has ended".
    Verified against
    src/ltl/ex/bpmn1.bpmn: `G ("A" -> X X "B")` and
    `G ("B" -> X FIREABLE(t_end_loop))` both hold there; the
    single-X/no-X versions (correct under "sticky_events") do not.
    """

    if event_semantics not in ("sticky_events", "event_per_state"):
        raise ValueError(
            f"Unknown event_semantics: {event_semantics!r} "
            "(expected 'event_per_state' or 'sticky_events')"
        )

    if verbose:
        print(
            f"    Places:      {len(net.places)}\n"
            f"    Transitions: {len(net.transitions)}"
        )

        print(f"[5] Adding {event_semantics}-semantics event places")

    extra_places: Dict[str, int] = {}
    extra_produce: Dict[str, List[Tuple[str, int]]] = {}
    suppress_produce: Set[str] = set()
    extra_transitions: Dict[str, Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]] = {}

    if event_semantics == "sticky_events":

        activity_symbols = create_activity_symbols(names, prefix="occ")

        extra_places.update({place_id: 0 for place_id in activity_symbols.values()})

        for transition in net.transitions:
            if transition.label is not None:
                occ_place = activity_symbols[transition.label]
                extra_produce.setdefault(transition.name, []).append((occ_place, 1))

    else:  # "event_per_state"

        activity_symbols = create_activity_symbols(names, prefix="evt")

        # Several distinct transitions can carry the same label (e.g. the
        # same task appearing twice in a BPMN model). They then share one
        # pulse place, so without more care any of their reset
        # transitions could consume a pulse emitted by another one of
        # them, sending the token to the wrong place -- which corrupts
        # the net's behaviour. The pulse place -- what formulas
        # observe -- is unchanged; labels used by a single transition
        # get no extra place.
        label_count: Dict[str, int] = {}
        for transition in net.transitions:
            if transition.label is not None:
                label_count[transition.label] = label_count.get(transition.label, 0) + 1

        for transition in net.transitions:

            if transition.label is None:
                continue

            tid = names.transitions[transition.name]
            evt_place = activity_symbols[transition.label]

            extra_places[evt_place] = 0
            suppress_produce.add(transition.name)
            extra_produce[transition.name] = [(evt_place, 1)]

            reset_consume = [(evt_place, 1)]

            if label_count[transition.label] > 1:
                ret_place = f"ret_{tid}"
                extra_places[ret_place] = 0
                extra_produce[transition.name].append((ret_place, 1))
                reset_consume.append((ret_place, 1))

            real_produce = [
                (names.places[arc.target.name], arc.weight)
                for arc in transition.out_arcs
            ]
            extra_transitions[f"reset_{tid}"] = (reset_consume, real_produce)

    if verbose:
        print(f"    Visible activities: {len(activity_symbols)}")

    # --------------------------------------------------------
    # LTL is defined over infinite paths, but a BPMN process is
    # finite -- it deadlocks once it reaches final_marking. Without a
    # successor there, LoLA's search for an infinite (accepting or
    # violating) run through that branch behaves as if the run simply
    # doesn't extend to satisfy any obligation depending on "the state
    # after this one" -- so formulas using X (nextstate), U, or any
    # obligation that only comes due at/after completion can give
    # wrong answers (verified: G(occ_X>=1 -> X(occ_X>=1)) -- trivially
    # true once occ_X is set, since occurrence counters are monotonic
    # -- is reported as VIOLATED without this fix). Add a transition
    # that only consumes/reproduces final_marking's own tokens, so
    # completed runs keep going forever instead of dead-ending.
    #
    # It must be a 2-cycle, not a single self-loop: LoLA (2.0) evaluates
    # X wrongly on a state whose only successor is itself (verified:
    # on a net ending in a self-loop, G(XX a -> XX a) is reported as
    # violated, and on the same net ending in a 2-cycle it holds). So
    # t_end_loop and t_end_loop_back alternate between two markings
    # that differ only in the tick places end_tick_a/end_tick_b; both
    # keep final_marking's own tokens, so anything that reads the final
    # marking (e.g. a place count) is stable across the cycle.
    #
    # This covers the intended, common case (the process legitimately
    # finishes); it does not add a self-loop to any *other*, unintended
    # deadlock the net might structurally have -- those remain finite
    # dead ends, and formulas sensitive to that can be checked directly
    # against LoLA's own DEADLOCK atomic proposition.
    #
    # (Under "event_per_state" semantics, final_marking's places are produced by
    # the last activity's *reset* transition rather than directly by
    # the activity's own transition -- that's fine, t_end_loop only
    # cares about the resulting place values, not who set them.)
    # --------------------------------------------------------

    final_marking_arcs = [
        (names.places[place.name], count)
        for place, count in final_marking.items()
        if count > 0
    ]

    end_condition = None

    if final_marking_arcs:
        # The final marking's tokens are kept in both states of the end
        # cycle (see above), so "the process has ended" is simply "the
        # final marking's places hold their tokens" -- a cheap place
        # comparison. Do not spell it FIREABLE(t_end_loop) OR
        # FIREABLE(t_end_loop_back): equivalent, but LoLA builds a much
        # larger Büchi automaton for it (measured: ~4x slower on a 875K
        # marking net).
        end_condition = "(" + " AND ".join(
            f"{place_id} >= {count}" for place_id, count in final_marking_arcs
        ) + ")"

        extra_places["end_tick_a"] = 1
        extra_places["end_tick_b"] = 0
        extra_transitions["t_end_loop"] = (
            final_marking_arcs + [("end_tick_a", 1)],
            final_marking_arcs + [("end_tick_b", 1)],
        )
        extra_transitions["t_end_loop_back"] = (
            final_marking_arcs + [("end_tick_b", 1)],
            final_marking_arcs + [("end_tick_a", 1)],
        )

    net_text = render_lola(
        net,
        initial_marking,
        names,
        extra_places=extra_places,
        extra_produce=extra_produce,
        extra_transitions=extra_transitions,
        suppress_produce=suppress_produce,
    )

    return LolaModel(
        net_text=net_text,
        names=names,
        activity_symbols=activity_symbols,
        end_condition=end_condition,
    )


# ============================================================
# 2. LTL constraint compiler
# ============================================================

def activity(activity_symbols: Dict[str, str], label: str) -> str:
    """Convert an activity label into a LoLA atomic proposition."""

    if label not in activity_symbols:

        available = "\n".join(f"    - {x}" for x in sorted(activity_symbols))
        
        raise ValueError(
            f"Activity '{label}' was not found.\nAvailable activities:\n{available}"
        )

    event_place = activity_symbols[label]

    return f"({event_place} >= 1)"


def substitute_activities(
    activity_symbols: Dict[str, str],
    formula: str,
    end_condition: Optional[str] = None,
) -> str:
    """
    Expand "double-quoted activity labels" inside a raw LoLA formula
    into their event-place atomic proposition, e.g.:

        G ("Take blood culture" -> F "Administer antibiotics")
        -> G ((occ_t1_... >= 1) -> F (occ_t4_... >= 1))

    The placeholder `@end` expands to `end_condition`, i.e. "the process
    has reached its final marking", e.g. `F ("A" AND XX @end)`. Prefer
    it to spelling the condition with FIREABLE(t_end_loop): the end of
    the net is a 2-cycle (t_end_loop / t_end_loop_back), so no single
    FIREABLE atom is true in every end state, and the OR of both is
    markedly slower to check than the place condition `@end` expands to.

    Quoting is required rather than matching bare label text, because
    LoLA reserves the single letters A/G/E/F/X/U/R (and runs thereof,
    e.g. "AG") as CTL*/LTL operators, plus keywords like AND/OR/NOT/
    FIREABLE -- exactly the kind of short names activities often have
    (our own Test1.bpmn fixture uses "A"/"B"/"C"/"D"). Since LoLA's own
    grammar never uses a literal `"` character, quoting an activity
    reference can't collide with real LoLA syntax.
    """

    if "@end" in formula:
        if end_condition is None:
            raise ValueError(
                "Formula uses @end, but the net has no final marking to "
                "define the end of the process"
            )
        formula = formula.replace("@end", end_condition)

    return re.sub(
        r'"([^"]+)"',
        lambda match: activity(activity_symbols, match.group(1)),
        formula,
    )


def _precedence_formula(a: str, b: str, event_semantics: str) -> str:
    """
    "B may occur only if A occurred previously", given `a`/`b` as
    already-built atomic propositions (from activity()).

    Under "sticky_events" semantics, a/b are monotonic (once true, always
    true), so "B implies A already happened" is exactly: whenever B
    has occurred, A has also occurred -- no past-time operator needed
    (LoLA's LTL has none).

    Under "event_per_state" semantics, a/b are true for exactly one state each,
    so that trick no longer applies -- instead this checks, once, over
    the whole trace: B's pulse never fires before A's has (a weak-until
    reading: "not B until A, or B never fires at all"). LoLA has no
    native weak-until, so it's spelled out via strong U:
    "(NOT B) U (A OR G NOT B)".

    It has to be ONE temporal formula. The more obvious
    "((NOT B) U A) OR (G NOT B)" is equivalent, but LoLA splits a
    top-level OR (or ->) between temporal formulas into two separate
    "holds on all paths" checks -- "every run satisfies (NOT B) U A, or
    every run satisfies G NOT B" -- instead of checking each run against
    the disjunction. That reports a violation whenever some runs contain
    A and others never contain B (verified on CAP: a branch that never
    performs A or B fails the strong U on that run and the other
    branch fails the G, so the OR of the two verdicts is "no"). The
    single-U form is checked per run, as intended.
    """

    if event_semantics == "event_per_state":

        not_b = f"NOT {b}"

        return f"(({not_b}) U ({a} OR G ({not_b})))"

    else:
        return f"G ({b} -> {a})"


def compile_constraint(
    activity_symbols: Dict[str, str],
    constraint: dict,
    event_semantics: str = "event_per_state",
    end_condition: Optional[str] = None,
) -> str:

    constraint_type = constraint["type"].lower()

    if constraint_type == "response":

        a = activity(activity_symbols, constraint["a"])
        b = activity(activity_symbols, constraint["b"])

        return f"G ({a} -> F {b})"

    elif constraint_type == "precedence":

        a = activity(activity_symbols, constraint["a"])
        b = activity(activity_symbols, constraint["b"])

        return _precedence_formula(a, b, event_semantics)

    elif constraint_type == "existence":

        a = activity(activity_symbols, constraint["a"])

        return f"F {a}"

    elif constraint_type == "absence":

        a = activity(activity_symbols, constraint["a"])

        return f"G NOT {a}"

    elif constraint_type == "succession":

        a = activity(activity_symbols, constraint["a"])
        b = activity(activity_symbols, constraint["b"])

        response = f"G ({a} -> F {b})"
        precedence = _precedence_formula(a, b, event_semantics)

        return f"({response}) AND ({precedence})"

    elif constraint_type == "raw":

        # Advanced escape hatch: an arbitrary LoLA formula, written
        # directly against LoLA identifiers (see the generated .lola
        # file / model.names for the place/transition mapping), or
        # against "double-quoted activity labels" (see
        # substitute_activities()) wherever an occurrence check is
        # wanted instead.

        return substitute_activities(
            activity_symbols, constraint["formula"], end_condition
        )

    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")


# ============================================================
# 3. Run LoLA
# ============================================================

def find_lola(lola_command: str) -> str:

    executable = shutil.which(lola_command)

    if executable is not None:
        return executable

    path = Path(lola_command)

    if path.exists():
        return str(path)

    raise RuntimeError(
        "LoLA was not found.\n\n"
        "Build it under src/ltl/lola/lola-2.0/ (see its INSTALL file), "
        "put it on PATH, or use:\n\n"
        "    --lola /path/to/lola"
    )


def run_lola(
    net_file: str,
    formula: str,
    lola_command: str,
) -> Tuple[bool, str, str]:
    """
    Run LoLA on `net_file` with `formula`. Returns (holds, witness,
    stderr): `holds` is True/False as parsed from LoLA's "result: yes"/
    "result: no" line (LoLA writes all of this to stderr).

    The witness path is requested via --path=<tempfile> rather than
    the bare -p (which defaults to stdout): this LoLA 2.0 build exits
    via _exit() instead of exit(), which skips flushing libc's stdio
    buffers, and its Output destructor deliberately never closes/
    flushes stdout itself -- so whenever stdout isn't a TTY (always
    true for subprocess.PIPE), any -p/-s/-j output written there is
    silently lost. Writing to a real file goes through fclose(),
    which does flush, so that's the only reliable way to retrieve it.
    """

    executable = find_lola(lola_command)

    with tempfile.NamedTemporaryFile(suffix=".path", delete=False) as tmp:
        path_file = tmp.name

    try:
        result = subprocess.run(
            [executable, net_file, f"--formula={formula}", f"--path={path_file}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        match = re.search(r"result:\s*(yes|no)", result.stderr, flags=re.IGNORECASE)

        if match is None:
            raise RuntimeError(
                "Could not parse LoLA's result from its output:\n\n" + result.stderr
            )

        holds = match.group(1).lower() == "yes"
        witness = Path(path_file).read_text(encoding="utf-8")

        return holds, witness, result.stderr
    finally:
        Path(path_file).unlink(missing_ok=True)


# ============================================================
# 4. Reporting
# ============================================================

def print_results(name: str, holds: bool, witness=None, error=None) -> None:
    out = ("[PASS]" if holds else "[FAIL]") + " "

    if not holds: 
        if witness is not None and witness.strip():
            out += "Witness path:"
            for line in witness.strip().splitlines():
                out += f"        {line}"
        elif error is not None:
            out += error

    print(out)


def print_model_summary(model: LolaModel) -> None:
    print()
    print("=" * 70)
    print("ACTIVITY MAPPING (activity -> event place)")
    print("=" * 70)

    for label, symbol in sorted(model.activity_symbols.items()):
        print(f"{label:<45} -> {symbol}")


# ============================================================
# 5. Main pipeline
# ============================================================

def _default_output_path(source_file: str, extension: str) -> str:
    """
    Derive a default generated-output path alongside `source_file`:
    same directory and stem, marked with a "_lola" suffix so it can't
    collide with `source_file` itself (relevant when `source_file` is
    itself a .pnml, or when it shares a stem with a hand-written .svg),
    plus `extension` (e.g. ".svg"). "model.bpmn" -> "model_lola.svg".
    """

    return str(Path(source_file).with_name(Path(source_file).stem + "_lola" + extension))


def check_constraints(
    source_file: str,
    constraints: Union[List[dict], str],
    lola_command: str = "lola",
    output_lola: Optional[str] = None,
    output_svg: Optional[str] = None,
    output_pnml: Optional[str] = None,
    verbose: bool = True,
    event_semantics: str = "event_per_state",
) -> LolaCheckResult:
    """
    Library entry point -- the single function client code (a web
    backend, a notebook, another script) should call to run this whole
    pipeline: convert `source_file` (.bpmn or .pnml) to a Petri net,
    render it to SVG, optionally export it as PNML, build the LoLA net
    (with event places, see `event_semantics`), and run each of
    `constraints` (see the CONSTRAINTS docstring above for the schema)
    through LoLA.

    `constraints` is either a list of constraint dicts, or a path
    (str) to a JSON/YAML file holding that same list -- see
    load_constraints_file(), which is used to load it in the latter
    case.

    `event_semantics` selects how "activity X occurred" is exposed to
    formulas -- see construct_model()'s docstring for the full
    explanation:

    - "event_per_state" (default): a genuine "just fired" flag
      (`evt_<t>`, true for exactly one state), giving real
      per-occurrence event-trace semantics -- response/existence/
      absence are exact regardless of how many times an activity can
      recur; precedence compiles to a different (also exact) formula
      under this mode.
    - "sticky_events": a monotonic per-activity counter (`occ_<t>`,
      means "X occurred at least once so far"). response/precedence/
      succession compile to formulas that are exact for activities
      firing at most once per instance and a sound approximation
      otherwise.

    Returns a LolaCheckResult with the generated model and, per constraint,
    its compiled formula, PASS/FAIL, and any witness path -- as data,
    not just console output. Unlike validate() (the CLI entry point,
    which calls this function and then exits non-zero on failure), this
    never calls sys.exit() and never raises for a failing constraint --
    only for genuine errors (bad input file, unknown activity, LoLA not
    found, ...). Pass verbose=False to suspend the progress/summary
    printing this function otherwise does.

    output_pnml=None (the default) skips PNML export entirely; pass an
    explicit path to also write one (skipped instead, with a note, if
    that path resolves to the same file as `source_file`).
    """

    if isinstance(constraints, str):
        constraints = load_constraints_file(constraints)

    if output_lola is None:
        output_lola = _default_output_path(source_file, ".lola")

    if output_svg is None:
        output_svg = _default_output_path(source_file, ".svg")

    net, initial_marking, final_marking = load_petri_net(source_file, verbose=verbose)

    names = assign_names(net)

    if verbose:
        print("[3] Rendering Petri net to SVG")

    visualize_petri_net(net, initial_marking, final_marking, output_svg, names)

    if verbose:
        print(f"    Written to {output_svg}")

    pnml_written: Optional[str] = None

    if output_pnml is not None:

        if Path(output_pnml).resolve() == Path(source_file).resolve():
            if verbose:
                print(f"[4] Skipping PNML export (input is already {source_file})")
        else:
            if verbose:
                print("[4] Exporting intermediate Petri net to PNML")

            save_petri_net_pnml(net, initial_marking, final_marking, output_pnml)
            pnml_written = output_pnml

            if verbose:
                print(f"    Written to {output_pnml}")

    model = construct_model(
        net,
        initial_marking,
        final_marking,
        names,
        verbose=verbose,
        event_semantics=event_semantics,
    )

    # if verbose:
    #     print_model_summary(model)

    Path(output_lola).write_text(model.net_text, encoding="utf-8")

    if verbose:
        print()
        print(f"[6] LoLA net written to {output_lola}")

    results: List[LolaConstraintResult] = []

    if not constraints:

        if verbose:
            print()
            print("WARNING: CONSTRAINTS is empty. Add at least one LTL constraint.")

        return LolaCheckResult(
            model=model,
            lola_path=output_lola,
            svg_path=output_svg,
            pnml_path=pnml_written,
            results=results,
        )

    if verbose:
        print("[7] Running LoLA")
        print()
        print("=" * 70)
        print("LTL VERIFICATION RESULTS")
        print("=" * 70)

    for constraint in constraints:
        try:
            formula = compile_constraint(
                model.activity_symbols, constraint, event_semantics,
                model.end_condition,
            )
        except Exception as e:
            if verbose:
                print(f"{constraint['name']}")
            formula = None; holds = False; witness = None; error=str(e)
        else:
            error = None
            if verbose:
                print(f"{constraint['name']}: {formula}")
            holds, witness, _log = run_lola(output_lola, formula, lola_command)

        if verbose:
            print_results(constraint["name"], holds, witness=witness, error=error)

        results.append(
            LolaConstraintResult(
                name=constraint["name"],
                formula=formula,
                holds=holds,
                witness=witness,
                error=error
            )
        )

    return LolaCheckResult(
        model=model,
        lola_path=output_lola,
        svg_path=output_svg,
        pnml_path=pnml_written,
        results=results,
    )


def validate(
    source_file: str,
    constraints: Union[List[dict], str],
    lola_command: str,
    output_lola: Optional[str] = None,
    output_svg: Optional[str] = None,
    output_pnml: Optional[str] = None,
    event_semantics: str = "event_per_state",
) -> None:
    """CLI entry point: run check_constraints() and exit(1) on any failing constraint."""

    result = check_constraints(
        source_file=source_file,
        constraints=constraints,
        lola_command=lola_command,
        output_lola=output_lola,
        output_svg=output_svg,
        output_pnml=output_pnml,
        verbose=True,
        event_semantics=event_semantics,
    )

    if not result.all_hold:
        sys.exit(1)


def load_constraints_file(path: str) -> List[dict]:
    """
    Load a list of constraint dicts (same schema as CONSTRAINTS above)
    from a JSON or YAML file, chosen by extension.
    """

    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")

    if file_path.suffix.lower() in (".yaml", ".yml"):

        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "Reading .yaml/.yml constraint files requires PyYAML "
                "(pip install pyyaml)."
            ) from exc

        data = yaml.safe_load(text)

    else:
        data = json.loads(text)

    if not isinstance(data, list):
        raise ValueError(
            f"{path}: expected a JSON/YAML list of constraint dicts, "
            f"got {type(data).__name__}"
        )

    return data


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Validate a BPMN/PNML process model against LTL constraints using LoLA."
    )

    parser.add_argument("model", help="Path to a .bpmn or .pnml file")

    parser.add_argument(
        "--lola",
        default="lola",
        help=(
            "LoLA executable or path "
            "(default: lola on PATH; falls back to "
            "src/ltl/lola/lola-2.0/src/lola if that literal path is passed)"
        ),
    )

    parser.add_argument(
        "--net",
        default=None,
        help=(
            "Generated LoLA net file "
            "(default: alongside the input model, same name with a "
            '"_lola" suffix and a .lola extension)'
        ),
    )

    parser.add_argument(
        "--svg",
        default=None,
        help=(
            "Petri net visualization file "
            "(default: alongside the input model, same name with a "
            '"_lola" suffix and a .svg extension)'
        ),
    )

    # parser.add_argument(
    #     "--pnml",
    #     default=None,
    #     help=(
    #         "Intermediate Petri net PNML file "
    #         "(default: alongside the input model, same name with a "
    #         ".pnml extension; skipped if the input is already PNML at "
    #         "that same path)"
    #     ),
    # )

    parser.add_argument(
        "--constraints",
        default=None,
        help=(
            "Path to a JSON/YAML file holding a list of constraint dicts "
            "(same schema as the CONSTRAINTS list in this file). Defaults "
            "to the CONSTRAINTS list hardcoded in this file."
        ),
    )

    parser.add_argument(
        "--event-semantics",
        choices=["event_per_state", "sticky_events"],
        default="event_per_state",
        help=(
            "How 'activity X occurred' is exposed to formulas: "
            "'event_per_state' (default) is a genuine per-firing flag, "
            "exact for every occurrence but changes how 'precedence' "
            "compiles; 'sticky_events' is a monotonic counter -- see "
            "construct_model()'s docstring."
        ),
    )

    args = parser.parse_args()

    constraints = (
        load_constraints_file(args.constraints)
        if args.constraints is not None
        else CONSTRAINTS
    )

    validate(
        source_file=args.model,
        constraints=constraints,
        lola_command=args.lola,
        output_lola=args.net,
        output_svg=args.svg,
        # output_pnml=args.pnml,
        output_pnml=None,
        event_semantics=args.event_semantics,
    )


if __name__ == "__main__":
    main()
