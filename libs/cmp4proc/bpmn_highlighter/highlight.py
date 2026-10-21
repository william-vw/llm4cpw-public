"""Load, validate, and apply highlight rules to a diagram's elements.

Two responsibilities live here:

- **Load + validate** (``load_highlight_config``) takes a plain ``dict``
  (already-deserialized JSON or YAML — the two are structurally equivalent)
  and turns it into a validated :class:`HighlightConfig`. This module never
  touches the filesystem; reading a ``.yaml``/``.json`` file and calling
  ``json.load``/``yaml.safe_load`` happens only in ``cli.py``. Highlight
  config is treated as untrusted input: every rule is fully validated
  (unknown match kinds, malformed colors, unknown symbol names, unknown
  keys) and a bad config raises :class:`HighlightConfigError` rather than
  rendering a silently-wrong diagram.

- **Apply** (``highlight_elements``) checks each rule against every element
  via its ``match`` selector (``id``, ``type``, or ``tag``). Rules are
  applied in list order and the **first matching rule wins** per element —
  later rules never override an element that an earlier rule already
  claimed. This keeps precedence simple and predictable (put more specific
  ``id`` rules before broader ``type`` rules) without needing boolean rule
  combinators; see the "Open questions" section in CLAUDE.md if that stops
  being sufficient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from .model import Diagram, DiagramElement

# 3/6/8-digit hex, optionally with an alpha channel (#rgb, #rrggbb, #rrggbbaa).
HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")

# The built-in symbol set the renderer knows how to draw. Keep in sync with
# the overlay drawing functions in renderer.py.
BUILTIN_SYMBOLS = frozenset(
    {"warning", "robot", "clock", "star", "check", "cross", "flag", "lock", "info", "add"}
)

MatchKind = Literal["id", "type", "tag"]
_VALID_MATCH_KINDS: frozenset[str] = frozenset({"id", "type", "tag"})
_KNOWN_RULE_KEYS = {"match", "color", "symbol", "label", "border"}


class HighlightConfigError(ValueError):
    """Raised when a highlight config dict fails validation."""


@dataclass(frozen=True, slots=True)
class MatchSelector:
    """A single-field selector: match by element id, BPMN type, or custom tag."""

    kind: MatchKind
    value: str


@dataclass(frozen=True, slots=True)
class HighlightRule:
    """One resolved, validated highlight rule."""

    match: MatchSelector
    color: str | None = None
    symbol: str | None = None
    label: str | None = None
    border: str | None = None
    rule_index: int = -1  # position in the source "highlights" list


@dataclass(frozen=True, slots=True)
class HighlightConfig:
    """A validated set of highlight rules, ready to apply to a diagram."""

    rules: list[HighlightRule] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ResolvedHighlight:
    """The highlight styling that applies to one diagram element."""

    element_id: str
    color: str | None
    symbol: str | None
    label: str | None
    border: str | None
    rule_index: int


def load_highlight_config(data: dict[str, Any], diagram: Diagram | None = None) -> HighlightConfig:
    """Validate a plain highlight-config dict into a :class:`HighlightConfig`.

    ``data`` is exactly the shape documented in CLAUDE.md's "Highlight config
    format" — the same dict regardless of whether it came from parsed YAML,
    parsed JSON, or was built up in code.

    If ``diagram`` is given, ``id`` selectors are checked against the
    diagram's actual element ids and an unknown id fails loudly. ``type`` and
    ``tag`` selectors are never "unknown" in that sense — they're filters
    that may simply match zero elements in a given diagram.
    """
    if not isinstance(data, dict):
        raise HighlightConfigError(f"highlight config must be a dict, got {type(data).__name__}")

    raw_rules = data.get("highlights")
    if raw_rules is None:
        raise HighlightConfigError('highlight config must have a top-level "highlights" list')
    if not isinstance(raw_rules, list):
        raise HighlightConfigError('"highlights" must be a list')

    rules = [_load_rule(raw_rule, i, diagram) for i, raw_rule in enumerate(raw_rules)]
    return HighlightConfig(rules=rules)


def _load_rule(raw_rule: Any, index: int, diagram: Diagram | None) -> HighlightRule:
    if not isinstance(raw_rule, dict):
        raise HighlightConfigError(
            f"highlights[{index}] must be a dict, got {type(raw_rule).__name__}"
        )

    match = _load_match_selector(raw_rule.get("match"), index, diagram)

    color = raw_rule.get("color")
    if color is not None and (not isinstance(color, str) or not HEX_COLOR_RE.match(color)):
        raise HighlightConfigError(
            f'highlights[{index}].color "{color}" is not a valid hex color '
            '(expected e.g. "#e74c3c")'
        )

    symbol = raw_rule.get("symbol")
    if symbol is not None and (not isinstance(symbol, str) or symbol not in BUILTIN_SYMBOLS):
        raise HighlightConfigError(
            f'highlights[{index}].symbol "{symbol}" is not a known symbol '
            f"(known: {sorted(BUILTIN_SYMBOLS)})"
        )

    label = raw_rule.get("label")
    if label is not None and not isinstance(label, str):
        raise HighlightConfigError(f"highlights[{index}].label must be a string")

    border = raw_rule.get("border")
    if border is not None and not isinstance(border, str):
        raise HighlightConfigError(f"highlights[{index}].border must be a string")

    unknown_keys = set(raw_rule) - _KNOWN_RULE_KEYS
    if unknown_keys:
        raise HighlightConfigError(
            f"highlights[{index}] has unknown key(s): {sorted(unknown_keys)}"
        )

    return HighlightRule(
        match=match, color=color, symbol=symbol, label=label, border=border, rule_index=index
    )


def _load_match_selector(raw_match: Any, index: int, diagram: Diagram | None) -> MatchSelector:
    if not isinstance(raw_match, dict) or not raw_match:
        raise HighlightConfigError(f'highlights[{index}] must have a non-empty "match" dict')

    match_kinds = [k for k in raw_match if k in _VALID_MATCH_KINDS]
    if len(raw_match) != 1 or not match_kinds:
        raise HighlightConfigError(
            f"highlights[{index}].match must have exactly one of "
            f"{sorted(_VALID_MATCH_KINDS)}, got {sorted(raw_match)}"
        )

    kind = match_kinds[0]
    value = raw_match[kind]
    if not isinstance(value, str) or not value:
        raise HighlightConfigError(f"highlights[{index}].match.{kind} must be a non-empty string")

    if diagram is not None and kind == "id" and value not in diagram.elements:
        raise HighlightConfigError(
            f'highlights[{index}].match.id "{value}" does not exist in the diagram'
        )

    return MatchSelector(kind=kind, value=value)


def highlight_elements(diagram: Diagram, config: HighlightConfig) -> dict[str, ResolvedHighlight]:
    """Return the resolved highlight for every element that matched a rule.

    Elements with no matching rule are simply absent from the result — the
    renderer draws them with default (non-highlighted) styling.
    """
    resolved: dict[str, ResolvedHighlight] = {}
    for element in diagram.elements.values():
        rule = _first_matching_rule(element, config.rules)
        if rule is not None:
            resolved[element.id] = ResolvedHighlight(
                element_id=element.id,
                color=rule.color,
                symbol=rule.symbol,
                label=rule.label,
                border=rule.border,
                rule_index=rule.rule_index,
            )
    return resolved


def _first_matching_rule(
    element: DiagramElement, rules: list[HighlightRule]
) -> HighlightRule | None:
    for rule in rules:
        if _selector_matches(element, rule.match):
            return rule
    return None


def _selector_matches(element: DiagramElement, selector: MatchSelector) -> bool:
    if selector.kind == "id":
        return element.id == selector.value
    if selector.kind == "type":
        return element.type == selector.value
    if selector.kind == "tag":
        return selector.value in element.tags
    return False
