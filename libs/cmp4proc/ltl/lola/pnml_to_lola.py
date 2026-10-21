"""
pnml_to_lola.py

Convert a Petri net -- loaded from a PNML file, or already available as a
pm4py PetriNet (e.g. produced by pm4py.convert_to_petri_net() from a BPMN
model) -- into LoLA 2.0's native low-level net format.

Why identifiers are rewritten
------------------------------
LoLA's net-format lexer accepts any run of characters except whitespace
and `,;:(){}` as an IDENTIFIER, EXCEPT that a purely-numeric token is
lexed as NUMBER instead, and the words PLACE/MARKING/TRANSITION/CONSUME/
PRODUCE/SAFE/FAIR/STRONG/WEAK are reserved keywords. PNML/BPMN element
names routinely contain spaces, punctuation or are purely numeric IDs
(e.g. "Task_1a2b3c" is fine, but a place named "1" or a label "Send
Invoice" is not). This module therefore always renames every place and
transition to a stable `p<i>_<hint>` / `t<i>_<hint>` identifier and
returns the original-name mapping alongside the generated text, so
callers can translate human-readable names into LoLA identifiers when
building formulas.

Usage as a library
-------------------
    from pnml_to_lola import petri_net_to_lola, pnml_file_to_lola

    net, im, fm = pm4py.convert_to_petri_net(bpmn)
    lola_text, names = petri_net_to_lola(net, im)

Usage as a CLI
--------------
    python pnml_to_lola.py model.pnml -o model.lola
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pm4py
from pm4py.objects.petri_net.obj import Marking, PetriNet

# ============================================================
# Identifier sanitization
# ============================================================

# Anything LoLA's IDENTIFIER token can't contain.
_FORBIDDEN = re.compile(r"[,;:()\{\}\s]")

# pm4py mints UUIDs for BPMN elements and uses them verbatim in the
# place/transition names of the converted Petri net (e.g. a task's
# wrapper places are named "ent__<uuid>"/"exi__<uuid>", and an
# unlabeled/silent transition is often just "_<uuid>"). Strip these out
# of hints -- they're unique but not remotely readable.
_UUID = re.compile(
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)


def _clean_hint(text: str) -> str:

    text = _UUID.sub("", str(text))
    text = _FORBIDDEN.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text


def _sanitize(prefix: str, text: str, index: int) -> str:
    """
    Build a LoLA-safe identifier. The stable `<prefix><index>` part
    guarantees uniqueness, guards against colliding with a keyword, and
    guards against being lexed as a NUMBER; a cleaned-up version of the
    original text is appended purely as a human-readable hint.
    """

    hint = _clean_hint(text)
    base = f"{prefix}{index}"

    return f"{base}_{hint}" if hint else base


@dataclass
class LolaNames:
    """
    Maps original pm4py place/transition names (and visible transition
    labels) to the sanitized identifiers used in the generated LoLA net.
    """

    places: Dict[str, str] = field(default_factory=dict)
    transitions: Dict[str, str] = field(default_factory=dict)
    # Visible (labeled) transitions only: BPMN activity label -> LoLA id.
    activities: Dict[str, str] = field(default_factory=dict)


def assign_names(net: PetriNet) -> LolaNames:
    """
    Assign sanitized LoLA identifiers to every place and transition of
    a pm4py PetriNet. Deterministic (sorted by original name) so the
    same net always maps to the same identifiers.
    """

    names = LolaNames()

    for i, place in enumerate(sorted(net.places, key=lambda p: str(p.name))):
        names.places[place.name] = _sanitize("p", place.name, i)

    for i, transition in enumerate(
        sorted(net.transitions, key=lambda t: str(t.name))
    ):
        # Prefer the BPMN activity label as the hint source -- it's
        # what a human actually recognizes -- and fall back to the
        # pm4py-internal name (UUID-stripped) for silent transitions.
        hint_source = (
            transition.label if transition.label is not None else transition.name
        )

        names.transitions[transition.name] = _sanitize("t", hint_source, i)

        if transition.label is not None:
            names.activities[transition.label] = names.transitions[transition.name]

    return names


# ============================================================
# Rendering
# ============================================================

def render_lola(
    net: PetriNet,
    initial_marking: Marking,
    names: LolaNames,
    extra_places: Optional[Dict[str, int]] = None,
    extra_produce: Optional[Dict[str, List[Tuple[str, int]]]] = None,
    extra_transitions: Optional[
        Dict[str, Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]]
    ] = None,
    suppress_produce: Optional[Set[str]] = None,
) -> str:
    """
    Render a pm4py PetriNet + initial marking as LoLA net-format text.

    `extra_places`/`extra_produce` let a caller graft observer places
    onto the net (see ltl_check.py's event places) without this
    function needing to know why: `extra_places` adds places (id ->
    initial marking, usually 0), and `extra_produce` appends additional
    arcs (place id, weight) to a transition's PRODUCE list, keyed by the
    transition's *original* pm4py name.

    `extra_transitions` adds wholly new transitions not tied to any
    pm4py Transition object (see ltl_check.py's terminal self-loop):
    {new transition id: (consume arcs, produce arcs)}, each arc list a
    list of (place id, weight) pairs.

    `suppress_produce` is a set of *original* pm4py transition names
    whose real PRODUCE arcs (from `transition.out_arcs`) should be
    omitted -- only `extra_produce`'s entries for that transition (if
    any) are rendered. This lets a caller "hollow out" a transition so
    its real effects can be moved onto a separate transition instead
    (see ltl_check.py's event_semantics="event_per_state" encoding,
    which diverts a transition's real outputs onto a "reset" transition
    that only fires after the original's own event pulse).

    The final marking is deliberately not part of LoLA's own format --
    LoLA determines reachability/deadlocks structurally during model
    checking, it doesn't take a target marking as input.
    """

    extra_places = extra_places or {}
    extra_produce = extra_produce or {}
    extra_transitions = extra_transitions or {}
    suppress_produce = suppress_produce or set()

    lines: List[str] = []

    # --------------------------------------------------------
    # PLACE
    # --------------------------------------------------------

    all_place_ids = list(names.places.values()) + list(extra_places.keys())

    lines.append("PLACE")
    lines.append("    " + ",\n    ".join(all_place_ids) + ";")
    lines.append("")

    # --------------------------------------------------------
    # MARKING
    # --------------------------------------------------------

    marked = [
        f"{names.places[place.name]}: {initial_marking[place]}"
        for place in sorted(initial_marking, key=lambda p: str(p.name))
        if initial_marking[place] > 0
    ]

    marked += [
        f"{place_id}: {count}"
        for place_id, count in extra_places.items()
        if count > 0
    ]

    lines.append("MARKING")
    lines.append(("    " + ",\n    ".join(marked) + ";") if marked else "    ;")
    lines.append("")

    # --------------------------------------------------------
    # TRANSITION ... CONSUME ... PRODUCE ...
    # --------------------------------------------------------

    for transition in sorted(net.transitions, key=lambda t: str(t.name)):

        tid = names.transitions[transition.name]

        consume = [
            f"{names.places[arc.source.name]}: {arc.weight}"
            for arc in sorted(transition.in_arcs, key=lambda a: str(a.source.name))
        ]

        produce = (
            []
            if transition.name in suppress_produce
            else [
                f"{names.places[arc.target.name]}: {arc.weight}"
                for arc in sorted(
                    transition.out_arcs, key=lambda a: str(a.target.name)
                )
            ]
        )

        produce += [
            f"{place_id}: {weight}"
            for place_id, weight in extra_produce.get(transition.name, [])
        ]

        lines.append(f"TRANSITION {tid}")
        lines.append("    CONSUME " + ", ".join(consume) + ";")
        lines.append("    PRODUCE " + ", ".join(produce) + ";")
        lines.append("")

    for tid, (consume_arcs, produce_arcs) in extra_transitions.items():

        consume = [f"{place_id}: {weight}" for place_id, weight in consume_arcs]
        produce = [f"{place_id}: {weight}" for place_id, weight in produce_arcs]

        lines.append(f"TRANSITION {tid}")
        lines.append("    CONSUME " + ", ".join(consume) + ";")
        lines.append("    PRODUCE " + ", ".join(produce) + ";")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def petri_net_to_lola(
    net: PetriNet,
    initial_marking: Marking,
) -> Tuple[str, LolaNames]:
    """Convenience wrapper: assign names, then render. See render_lola()."""

    names = assign_names(net)
    return render_lola(net, initial_marking, names), names


def pnml_file_to_lola(
    pnml_path: str,
) -> Tuple[str, LolaNames, Marking, Marking]:
    """Load a PNML file via pm4py and convert it to LoLA format."""

    net, initial_marking, final_marking = pm4py.read_pnml(
        pnml_path, auto_guess_final_marking=True
    )

    text, names = petri_net_to_lola(net, initial_marking)

    return text, names, initial_marking, final_marking


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Convert a PNML Petri net to LoLA 2.0 net format."
    )

    parser.add_argument("pnml", help="Path to a PNML file")
    parser.add_argument(
        "-o", "--output", help="Output .lola file (default: print to stdout)"
    )

    args = parser.parse_args()

    text, names, _initial_marking, _final_marking = pnml_file_to_lola(args.pnml)

    if args.output:

        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")

    else:
        print(text)

    print("\nPlace mapping:")
    for original, lola_id in names.places.items():
        print(f"    {original!r:<40} -> {lola_id}")

    print("\nTransition mapping:")
    for original, lola_id in names.transitions.items():
        print(f"    {original!r:<40} -> {lola_id}")


if __name__ == "__main__":
    main()
