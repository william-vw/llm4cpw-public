"""Internal diagram model.

Everything downstream of parsing (matching, rendering) works against these
plain dataclasses rather than the lxml tree, so the rest of the pipeline
never needs to know how the BPMN XML/DI was structured.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Point:
    """A single x/y coordinate, as used by sequence-flow waypoints."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Bounds:
    """Position and size of a shape, taken from ``bpmndi:BPMNBounds``."""

    x: float
    y: float
    width: float
    height: float


@dataclass(slots=True)
class DiagramElement:
    """One semantic BPMN element (task, gateway, event, flow, pool, lane, ...).

    ``type`` is the normalized, tag-local element kind (e.g. ``"task"``,
    ``"exclusiveGateway"``, ``"sequenceFlow"``, ``"pool"``, ``"lane"``) used
    for type-based highlight matching. ``bpmn_type`` preserves the raw
    element tag name as it appeared in the source XML for cases where the
    normalized ``type`` loses information.

    Exactly one of ``bounds`` (shapes: tasks, events, gateways, pools,
    lanes, ...) or ``waypoints`` (edges: sequence/message/association flows)
    is expected to be set, depending on the element kind; the other stays
    at its default.
    """

    id: str
    type: str
    bpmn_type: str
    label: str | None = None
    bounds: Bounds | None = None
    waypoints: list[Point] = field(default_factory=list)
    parent_id: str | None = None
    source_ref: str | None = None
    target_ref: str | None = None
    tags: set[str] = field(default_factory=set)
    attrs: dict[str, str] = field(default_factory=dict)

    @property
    def is_shape(self) -> bool:
        return self.bounds is not None

    @property
    def is_edge(self) -> bool:
        return bool(self.waypoints)


@dataclass(slots=True)
class Diagram:
    """A fully parsed BPMN diagram: flat element map plus source metadata.

    ``source_bytes`` holds the raw BPMN XML captured at parse time,
    regardless of what form the caller originally handed to
    ``parse_diagram()`` (path, str/bytes, file-like, or an already-parsed
    lxml tree). Nothing in the core pipeline (matching, the default
    renderer) needs it — it exists so a renderer backend that requires a
    real BPMN file on disk (e.g. one delegating to an external tool that
    only accepts a file path) can reconstruct one without the caller having
    to keep their original source around.
    """

    elements: dict[str, DiagramElement] = field(default_factory=dict)
    source_name: str | None = None
    source_bytes: bytes | None = None

    def get(self, element_id: str) -> DiagramElement | None:
        return self.elements.get(element_id)

    def by_type(self, type_: str) -> list[DiagramElement]:
        return [el for el in self.elements.values() if el.type == type_]

    def shapes(self) -> list[DiagramElement]:
        return [el for el in self.elements.values() if el.is_shape]

    def edges(self) -> list[DiagramElement]:
        return [el for el in self.elements.values() if el.is_edge]
