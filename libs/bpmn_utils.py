#!/usr/bin/env python3
"""
Add bpmn:incoming / bpmn:outgoing child elements to a BPMN file.

For every <bpmn:sequenceFlow id=F sourceRef=S targetRef=T> the script:
  * adds  <bpmn:outgoing>F</bpmn:outgoing>  to the element whose id == S
  * adds  <bpmn:incoming>F</bpmn:incoming>  to the element whose id == T

Usage:
    python add_incoming_outgoing.py input.bpmn [output.bpmn]

If no output path is given, "<input>.fix.bpmn" is written next to the input.
Requires lxml:  pip install lxml
"""

import sys
from lxml import etree

# The BPMN 2.0 "MODEL" namespace that the bpmn: prefix maps to.
BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"

# Diagram-interchange namespaces used by add_dummy_di().
BPMNDI_NS = "http://www.omg.org/spec/BPMN/20100524/DI"
DI_NS = "http://www.omg.org/spec/DD/20100524/DI"
DC_NS = "http://www.omg.org/spec/DD/20100524/DC"

TASK_TYPES = {
    "task", "userTask", "serviceTask", "scriptTask", "businessRuleTask",
    "manualTask", "sendTask", "receiveTask", "subProcess", "callActivity",
    "transaction", "adHocSubProcess",
}
EVENT_TYPES = {
    "startEvent", "endEvent", "intermediateCatchEvent",
    "intermediateThrowEvent", "boundaryEvent",
}
GATEWAY_TYPES = {
    "exclusiveGateway", "parallelGateway", "inclusiveGateway",
    "eventBasedGateway", "complexGateway",
}
DATA_TYPES = {"dataObjectReference", "dataStoreReference"}
FLOW_TYPES = {"sequenceFlow", "messageFlow", "association"}

# Standard BPMN shape sizes (width, height), keyed by local element tag name.
_SHAPE_SIZES = {
    **{t: (100, 80) for t in TASK_TYPES},
    **{t: (36, 36) for t in EVENT_TYPES},
    **{t: (50, 50) for t in GATEWAY_TYPES},
    **{t: (36, 50) for t in DATA_TYPES},
}

_GRID_COLUMNS = 6
_COLUMN_SPACING = 150
_ROW_SPACING = 120


def _local_name(el):
    return etree.QName(el).localname


def _add_sub(parent, kind, flow_id):
    """Add <bpmn:{kind}>flow_id</bpmn:{kind}> to parent, unless it's already there."""
    tag = f"{{{BPMN_NS}}}{kind}"
    for existing in parent.findall(tag):
        if existing.text == flow_id:
            return False
    el = etree.SubElement(parent, tag)
    el.text = flow_id
    return True


def add_incoming_outgoing(in_path, out_path):
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(in_path, parser)

    root = tree.getroot()

    by_id = {}
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        el_id = el.get("id")
        if el_id is not None:
            by_id.setdefault(el_id, el)

    for flow in root.iter(f"{{{BPMN_NS}}}sequenceFlow"):
        flow_id = flow.get("id")
        source_ref = flow.get("sourceRef")
        target_ref = flow.get("targetRef")

        if not flow_id:
            print("sequenceFlow without id, skipped")
            continue

        source = by_id.get(source_ref)
        if source is None:
            print(f"{flow_id}: sourceRef '{source_ref}' not found")
        else:
            _add_sub(source, "outgoing", flow_id)

        target = by_id.get(target_ref)
        if target is None:
            print(f"{flow_id}: targetRef '{target_ref}' not found")
        else:
            _add_sub(target, "incoming", flow_id)

    tree.write(out_path, xml_declaration=True, encoding="UTF-8", pretty_print=True)


def add_dummy_di(in_path, out_path, columns=_GRID_COLUMNS):
    """Add a placeholder bpmndi:BPMNDiagram section to a DI-less BPMN file.

    This does not compute a real layout - it walks the top-level flow nodes
    of <bpmn:process> in document order and places them on a naive grid
    (fixed spacing, standard size per element type), then connects
    sequenceFlow/messageFlow/association edges with a straight 2-point
    waypoint from the source shape's right edge to the target shape's left
    edge. That's enough for DI-aware consumers (which otherwise skip any
    element without bounds/waypoints) to render something, without claiming
    to be a meaningful diagram layout.

    If the file already has a BPMNPlane, it's left untouched (nothing is
    written twice) and the function returns False. Nested elements (e.g.
    inside an expanded subProcess) and pool/lane containers are not laid
    out - this only covers a single top-level process.
    """
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(in_path, parser)
    root = tree.getroot()

    if root.find(f".//{{{BPMNDI_NS}}}BPMNPlane") is not None:
        print("BPMN file already has a BPMNPlane, skipped")
        tree.write(out_path, xml_declaration=True, encoding="UTF-8", pretty_print=True)
        return False

    process = root.find(f"{{{BPMN_NS}}}process")
    if process is None:
        raise ValueError("no bpmn:process element found")

    shape_nodes = [el for el in process if _local_name(el) in _SHAPE_SIZES]
    edge_nodes = [el for el in process if _local_name(el) in FLOW_TYPES]

    bounds_by_id = {}
    for i, el in enumerate(shape_nodes):
        width, height = _SHAPE_SIZES[_local_name(el)]
        col, row = i % columns, i // columns
        bounds_by_id[el.get("id")] = (col * _COLUMN_SPACING, row * _ROW_SPACING, width, height)

    diagram = etree.SubElement(
        root,
        f"{{{BPMNDI_NS}}}BPMNDiagram",
        nsmap={"bpmndi": BPMNDI_NS, "di": DI_NS, "dc": DC_NS},
    )
    diagram.set("id", "BPMNDiagram_dummy")

    plane = etree.SubElement(diagram, f"{{{BPMNDI_NS}}}BPMNPlane")
    plane.set("id", "BPMNPlane_dummy")
    plane.set("bpmnElement", process.get("id"))

    for el in shape_nodes:
        el_id = el.get("id")
        x, y, width, height = bounds_by_id[el_id]
        shape = etree.SubElement(plane, f"{{{BPMNDI_NS}}}BPMNShape")
        shape.set("id", f"{el_id}_di")
        shape.set("bpmnElement", el_id)
        bounds_el = etree.SubElement(shape, f"{{{DC_NS}}}Bounds")
        bounds_el.set("x", str(x))
        bounds_el.set("y", str(y))
        bounds_el.set("width", str(width))
        bounds_el.set("height", str(height))

    for el in edge_nodes:
        el_id = el.get("id")
        source_bounds = bounds_by_id.get(el.get("sourceRef"))
        target_bounds = bounds_by_id.get(el.get("targetRef"))
        if source_bounds is None or target_bounds is None:
            print(f"{el_id}: sourceRef/targetRef not laid out, edge skipped")
            continue

        sx, sy, sw, sh = source_bounds
        tx, ty, tw, th = target_bounds
        edge = etree.SubElement(plane, f"{{{BPMNDI_NS}}}BPMNEdge")
        edge.set("id", f"{el_id}_di")
        edge.set("bpmnElement", el_id)
        wp1 = etree.SubElement(edge, f"{{{DI_NS}}}waypoint")
        wp1.set("x", str(sx + sw))
        wp1.set("y", str(sy + sh / 2))
        wp2 = etree.SubElement(edge, f"{{{DI_NS}}}waypoint")
        wp2.set("x", str(tx))
        wp2.set("y", str(ty + th / 2))

    tree.write(out_path, xml_declaration=True, encoding="UTF-8", pretty_print=True)
    return True