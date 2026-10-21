"""BPMN XML (+ DI) -> internal :class:`~bpmn_highlighter.model.Diagram`.

Accepts whatever form the caller already has the diagram in — a filesystem
path, raw XML as ``str``/``bytes``, a file-like object, or an
already-parsed ``lxml.etree._Element``/``_ElementTree`` — so a client that
loaded the diagram itself never has to round-trip through a temp file.

Namespaces are always resolved by URI, never by assuming a literal prefix
like ``bpmn:`` or ``bpmndi:`` — real-world exports vary their prefixes.
"""

from __future__ import annotations

import os
from typing import IO

from lxml import etree

from .model import Bounds, Diagram, DiagramElement, Point

BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
BPMNDI_NS = "http://www.omg.org/spec/BPMN/20100524/DI"
DI_NS = "http://www.omg.org/spec/DD/20100524/DI"
DC_NS = "http://www.omg.org/spec/DD/20100524/DC"

# Local (namespace-stripped) tag names this pipeline treats as highlightable
# flow-node / flow elements, taken from the semantic ``bpmn:process`` tree.
TASK_TYPES = {
    "task",
    "userTask",
    "serviceTask",
    "scriptTask",
    "businessRuleTask",
    "manualTask",
    "sendTask",
    "receiveTask",
    "subProcess",
    "callActivity",
    "transaction",
    "adHocSubProcess",
}
EVENT_TYPES = {
    "startEvent",
    "endEvent",
    "intermediateCatchEvent",
    "intermediateThrowEvent",
    "boundaryEvent",
}
GATEWAY_TYPES = {
    "exclusiveGateway",
    "parallelGateway",
    "inclusiveGateway",
    "eventBasedGateway",
    "complexGateway",
}
FLOW_TYPES = {"sequenceFlow", "messageFlow", "association"}
DATA_TYPES = {"dataObjectReference", "dataStoreReference"}
SEMANTIC_TYPES = TASK_TYPES | EVENT_TYPES | GATEWAY_TYPES | FLOW_TYPES | DATA_TYPES

# Container elements that don't sit in the flow-node graph but still get
# their own shape in the DI and their own highlight-able identity.
CONTAINER_TYPE_MAP = {"participant": "pool", "lane": "lane"}

BpmnSource = (
    str | bytes | os.PathLike[str] | IO[bytes] | IO[str] | etree._Element | etree._ElementTree
)

_XML_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)


def _qname(ns: str, local: str) -> str:
    return f"{{{ns}}}{local}"


def _local_name(elem: etree._Element) -> str:
    return etree.QName(elem).localname


def _looks_like_xml(text: str) -> bool:
    return text.lstrip().startswith("<")


def _load_root(source: BpmnSource) -> tuple[etree._Element, str | None]:
    """Normalize any accepted input shape to a parsed root element."""
    if isinstance(source, etree._ElementTree):
        return source.getroot(), None
    if isinstance(source, etree._Element):
        return source, None
    if isinstance(source, (bytes, bytearray)):
        return etree.fromstring(bytes(source), parser=_XML_PARSER), None
    if isinstance(source, os.PathLike):
        path = os.fspath(source)
        return etree.parse(path, parser=_XML_PARSER).getroot(), path
    if isinstance(source, str):
        if _looks_like_xml(source):
            return etree.fromstring(source.encode("utf-8"), parser=_XML_PARSER), None
        return etree.parse(source, parser=_XML_PARSER).getroot(), source
    if hasattr(source, "read"):
        data = source.read()
        name = getattr(source, "name", None)
        if isinstance(data, str):
            data = data.encode("utf-8")
        return etree.fromstring(data, parser=_XML_PARSER), name
    raise TypeError(
        "bpmn_source must be a path, str/bytes of XML, a file-like object, or an "
        f"lxml Element/ElementTree; got {type(source)!r}"
    )


def parse_diagram(bpmn_source: BpmnSource) -> Diagram:
    """Parse a BPMN 2.0 document into the internal :class:`Diagram` model.

    Never touches anything but its input argument — no filesystem access
    beyond reading the given path (if a path was given), no network access.
    """
    root, source_name = _load_root(bpmn_source)

    # Index every element in the document that carries an id, so DI shapes
    # can be resolved back to their semantic element even for element kinds
    # this parser doesn't specifically model (groups, text annotations, ...).
    id_index: dict[str, etree._Element] = {}
    for el in root.iter():
        el_id = el.get("id")
        if el_id is not None:
            id_index[el_id] = el

    elements: dict[str, DiagramElement] = {}
    _collect_semantic_elements(root, elements)

    plane = root.find(f".//{_qname(BPMNDI_NS, 'BPMNPlane')}")
    if plane is not None:
        _apply_di(plane, elements, id_index)

    source_bytes = etree.tostring(root)
    return Diagram(elements=elements, source_name=source_name, source_bytes=source_bytes)


def _collect_semantic_elements(root: etree._Element, elements: dict[str, DiagramElement]) -> None:
    for el in root.iter():
        ln = _local_name(el)
        el_id = el.get("id")
        if el_id is None:
            continue

        if ln in SEMANTIC_TYPES:
            parent = el.getparent()
            parent_id = parent.get("id") if parent is not None else None
            elements[el_id] = DiagramElement(
                id=el_id,
                type=ln,
                bpmn_type=ln,
                label=el.get("name") or None,
                source_ref=el.get("sourceRef"),
                target_ref=el.get("targetRef"),
                parent_id=parent_id,
            )
        elif ln in CONTAINER_TYPE_MAP:
            elements[el_id] = DiagramElement(
                id=el_id,
                type=CONTAINER_TYPE_MAP[ln],
                bpmn_type=ln,
                label=el.get("name") or None,
            )


def _element_for_di_ref(
    ref_id: str,
    elements: dict[str, DiagramElement],
    id_index: dict[str, etree._Element],
) -> DiagramElement:
    """Get (creating if necessary) the DiagramElement a DI shape/edge refers to."""
    existing = elements.get(ref_id)
    if existing is not None:
        return existing

    src_el = id_index.get(ref_id)
    if src_el is not None:
        ln = _local_name(src_el)
        el_type = CONTAINER_TYPE_MAP.get(ln, ln)
        new_el = DiagramElement(
            id=ref_id, type=el_type, bpmn_type=ln, label=src_el.get("name") or None
        )
    else:
        # DI references an id we can't resolve in the source tree at all —
        # keep it visible (typed "unknown") rather than silently dropping it.
        new_el = DiagramElement(id=ref_id, type="unknown", bpmn_type="unknown")

    elements[ref_id] = new_el
    return new_el


def _apply_di(
    plane: etree._Element,
    elements: dict[str, DiagramElement],
    id_index: dict[str, etree._Element],
) -> None:
    for shape in plane.iter(_qname(BPMNDI_NS, "BPMNShape")):
        ref_id = shape.get("bpmnElement")
        if not ref_id:
            continue
        bounds_el = shape.find(_qname(DC_NS, "Bounds"))
        if bounds_el is None:
            continue
        el = _element_for_di_ref(ref_id, elements, id_index)
        el.bounds = Bounds(
            x=float(bounds_el.get("x", 0)),
            y=float(bounds_el.get("y", 0)),
            width=float(bounds_el.get("width", 0)),
            height=float(bounds_el.get("height", 0)),
        )

    for edge in plane.iter(_qname(BPMNDI_NS, "BPMNEdge")):
        ref_id = edge.get("bpmnElement")
        if not ref_id:
            continue
        waypoints = [
            Point(x=float(wp.get("x", 0)), y=float(wp.get("y", 0)))
            for wp in edge.findall(_qname(DI_NS, "waypoint"))
        ]
        if not waypoints:
            continue
        el = _element_for_di_ref(ref_id, elements, id_index)
        el.waypoints = waypoints
