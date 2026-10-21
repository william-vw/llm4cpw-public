"""Second SVG rendering backend, built on pm4py + Graphviz instead of hand-drawn SVG.

Where ``renderer.py`` draws every shape and flow itself from the DI geometry
captured by ``parser.py``, this module outsources the actual drawing to
pm4py's classic BPMN visualizer (``pm4py.visualization.bpmn``), which lays
the diagram out with Graphviz's ``dot`` engine rather than using the
diagram's original DI bounds/waypoints. :func:`bpmn_to_svg` is that
outsourced call, kept in its plain/reusable form; :func:`render_svg` is this
backend's counterpart to ``renderer.render_svg`` — it calls the same pm4py
building blocks and then overlays ``resolved`` highlights on top of the
Graphviz SVG output.

Requires the optional ``pm4py`` extra (``pip install -e ".[pm4py]"``), plus
the Graphviz ``dot`` binary on ``PATH`` — neither is a hard dependency of
the package, so importing this module is what pulls them in, not importing
``bpmn_highlighter`` itself.

Highlight matching here is inherently a best-effort id-mapping trick, not a
first-class feature of pm4py's visualizer:

- Graphviz's classic visualizer names each dot node after Python's
  ``id()`` of the in-memory pm4py node object, not the original BPMN XML
  ``id``. This module rebuilds that mapping from the pm4py-imported
  ``BPMN`` graph (``{str(id(node)): node.get_id()}``) and resolves it
  against each SVG node's ``<title>``, which Graphviz always sets to the
  dot node name.
- A flow's own id is dropped entirely by pm4py's visualizer -- edges are
  drawn from (source node, target node) pairs only. Flows are matched here
  by that (source id, target id) pair instead; if a diagram has more than
  one flow between the same two elements, only the first unmatched one is
  resolved per pair -- a Graphviz/pm4py limitation this module can't see
  around.
- Only color and border highlighting are supported on flows -- placing a
  symbol/label badge at a flow's midpoint would require parsing the
  arbitrary Bezier spline Graphviz emits in the edge's ``path`` attribute,
  which isn't attempted here.

Known gaps versus ``renderer.py``, inherent to outsourcing the drawing to
pm4py's classic visualizer (called here with swimlanes disabled, matching
the reference ``bpmn_to_svg`` this module is built from): pools/lanes are
never drawn as containers, so pool/lane highlights are a no-op, and any
node kind pm4py's visualizer doesn't draw (data objects/stores, ...) can't
be highlighted either since it never appears in the output.

Beyond highlights, this module also always captions two things pm4py's
plain ``classic.apply`` drops on the floor for every diagram, highlighted
or not: a gateway's actual BPMN ``name`` (``classic.apply`` only ever draws
the fixed "X"/"+"/"O"/"E" glyph, never the name) and a sequence/message
flow's condition ``name`` (e.g. "Yes"/"No"), using the same node/flow
id-map described above.
"""

from __future__ import annotations

import os
import re
import tempfile

import graphviz
from lxml import etree
from pm4py.objects.bpmn.importer import importer as bpmn_importer
from pm4py.objects.bpmn.obj import BPMN
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.bpmn.variants import classic

from .highlight import ResolvedHighlight
from .model import Diagram
from .parser import GATEWAY_TYPES
from .renderer import (
    BORDER_DASH_STYLES,
    DEFAULT_TEXT_COLOR,
    SYMBOL_GLYPHS,
    _midpoint,
    _truncate,
    _wrap_label,
)

SVG_NS = "http://www.w3.org/2000/svg"
_COORD_PAIR_RE = re.compile(r"(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)")


def bpmn_to_svg(bpmn_path: str, svg_path: str | None = None) -> graphviz.Digraph:
    """Render ``bpmn_path`` to SVG via pm4py's classic (Graphviz) BPMN visualizer."""
    gviz, _ = _classic_gviz_and_model(bpmn_path)

    if svg_path is not None:
        bpmn_visualizer.save(gviz, svg_path)

    return gviz


def render_svg(
    diagram: Diagram, resolved: dict[str, ResolvedHighlight], show_ids: bool = False
) -> str:
    """Render ``diagram`` to an SVG string via pm4py, with ``resolved`` highlights
    overlaid on top, plus (regardless of highlighting) each gateway's real name
    and each flow's condition label captioned in. See the module docstring for
    how all of this is matched and its known gaps versus
    :func:`bpmn_highlighter.renderer.render_svg`.

    ``show_ids``, when true, captions every task/gateway/event with its
    element id in small text above the shape. This is a post-render overlay
    rather than baked into the node's own label, since pm4py's classic
    visualizer hardcodes start/end event labels blank and always draws a
    gateway's fixed glyph instead of its name -- there is no label to append
    the id to for those element kinds, so tasks use the same overlay
    treatment for consistency.
    """
    path, temp_path = _resolve_bpmn_path(diagram)
    try:
        gviz, bpmn_model = _classic_gviz_and_model(path)
    finally:
        if temp_path is not None:
            os.remove(temp_path)

    svg_root = etree.fromstring(gviz.pipe(format="svg"))
    graph_group = svg_root.find(f".//{{{SVG_NS}}}g[@class='graph']")

    node_bpmn_id = {str(id(n)): n.get_id() for n in bpmn_model.get_nodes()}
    flow_ids_by_dot_pair: dict[tuple[str, str], list[str]] = {}
    for flow in bpmn_model.get_flows():
        key = (str(id(flow.get_source())), str(id(flow.get_target())))
        flow_ids_by_dot_pair.setdefault(key, []).append(flow.get_id())

    overlays: list[etree._Element] = []
    used_flow_ids: set[str] = set()

    for group in svg_root.iter(f"{{{SVG_NS}}}g"):
        group_class = group.get("class")
        if group_class == "node":
            dot_id = _title_text(group)
            bpmn_id = node_bpmn_id.get(dot_id) if dot_id else None
            element = diagram.get(bpmn_id) if bpmn_id else None
            highlight = resolved.get(bpmn_id) if bpmn_id else None
            gateway_name = (
                element.label if element is not None and element.type in GATEWAY_TYPES else None
            )

            bbox = None
            if highlight is not None:
                bbox = _apply_node_style(group, highlight)
            elif gateway_name is not None or show_ids:
                shape = _node_shape(group)
                bbox = _shape_bbox(shape) if shape is not None else None
            if bbox is None:
                continue

            if gateway_name is not None:
                overlays.append(_gateway_name_caption(bbox, gateway_name))
            if highlight is not None:
                overlays.extend(_node_overlays(bbox, highlight))
            if show_ids and bpmn_id is not None:
                overlays.append(_element_id_caption(bbox, bpmn_id))
        elif group_class == "edge":
            title = _title_text(group)
            if not title or "->" not in title:
                continue
            src_dot, _, tgt_dot = title.partition("->")
            candidates = flow_ids_by_dot_pair.get((src_dot, tgt_dot), [])
            flow_id = next((fid for fid in candidates if fid not in used_flow_ids), None)
            if flow_id is None:
                continue
            used_flow_ids.add(flow_id)

            highlight = resolved.get(flow_id)
            if highlight is not None:
                _apply_edge_style(group, highlight)

            flow_element = diagram.get(flow_id)
            if flow_element is not None and flow_element.label:
                path_el = group.find(f"{{{SVG_NS}}}path")
                points = _COORD_PAIR_RE.findall(path_el.get("d", "")) if path_el is not None else []
                if points:
                    midpoint = _midpoint([(float(x), float(y)) for x, y in points])
                    overlays.append(_flow_condition_label(midpoint, flow_element.label))

    if graph_group is not None:
        for overlay in overlays:
            graph_group.append(overlay)

    return str(etree.tostring(svg_root, encoding="unicode"))


def _classic_gviz_and_model(bpmn_path: str) -> tuple[graphviz.Digraph, BPMN]:
    bpmn_model = bpmn_importer.apply(bpmn_path)
    parameters = bpmn_visualizer.Variants.CLASSIC.value.Parameters
    gviz = classic.apply(
        bpmn_model, parameters={parameters.FORMAT: "svg", parameters.ENABLE_SWIMLANES: False}
    )
    return gviz, bpmn_model


def _resolve_bpmn_path(diagram: Diagram) -> tuple[str, str | None]:
    """Return (path, temp-path-to-delete-afterward-or-None) for pm4py's importer.

    pm4py's ``bpmn_importer`` needs a real file on disk. A diagram parsed
    from an existing path reuses it directly; otherwise this falls back to
    the raw XML captured at parse time (``Diagram.source_bytes``), written
    to a throwaway temp file the caller must remove once done with it.
    """
    if diagram.source_name and os.path.isfile(diagram.source_name):
        return diagram.source_name, None
    if diagram.source_bytes is None:
        raise ValueError(
            "diagram has no on-disk source and no captured XML to fall back to -- "
            "the pm4py-backed renderer needs a real .bpmn file"
        )
    fd, temp_path = tempfile.mkstemp(suffix=".bpmn")
    with os.fdopen(fd, "wb") as f:
        f.write(diagram.source_bytes)
    return temp_path, temp_path


def _title_text(group: etree._Element) -> str | None:
    title = group.find(f"{{{SVG_NS}}}title")
    return title.text if title is not None else None


def _node_shape(node_group: etree._Element) -> etree._Element | None:
    for tag in ("ellipse", "polygon"):
        shape = node_group.find(f"{{{SVG_NS}}}{tag}")
        if shape is not None:
            return shape
    return None


def _apply_node_style(
    node_group: etree._Element, highlight: ResolvedHighlight
) -> tuple[float, float, float, float] | None:
    """Recolor the node's shape in place; return its (x, y, width, height) bbox."""
    shape = _node_shape(node_group)
    if shape is None:
        return None
    if highlight.color:
        shape.set("fill", highlight.color)
        shape.set("stroke", highlight.color)
    dasharray = _dasharray(highlight)
    if dasharray:
        shape.set("stroke-dasharray", dasharray)
    return _shape_bbox(shape)


def _apply_edge_style(edge_group: etree._Element, highlight: ResolvedHighlight) -> None:
    path = edge_group.find(f"{{{SVG_NS}}}path")
    if path is not None and highlight.color:
        path.set("stroke", highlight.color)
    dasharray = _dasharray(highlight)
    if path is not None and dasharray:
        path.set("stroke-dasharray", dasharray)

    arrowhead = edge_group.find(f"{{{SVG_NS}}}polygon")
    if arrowhead is not None and highlight.color:
        arrowhead.set("fill", highlight.color)
        arrowhead.set("stroke", highlight.color)


def _dasharray(highlight: ResolvedHighlight) -> str | None:
    if not highlight.border:
        return None
    return BORDER_DASH_STYLES.get(highlight.border, highlight.border)


def _shape_bbox(shape: etree._Element) -> tuple[float, float, float, float] | None:
    tag = etree.QName(shape).localname
    if tag == "ellipse":
        cx, cy = float(shape.get("cx", 0)), float(shape.get("cy", 0))
        rx, ry = float(shape.get("rx", 0)), float(shape.get("ry", 0))
        return cx - rx, cy - ry, 2 * rx, 2 * ry

    points_attr = shape.get("points")
    if not points_attr:
        return None
    points = [tuple(float(v) for v in p.split(",")) for p in points_attr.split() if p]
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _node_overlays(
    bbox: tuple[float, float, float, float], highlight: ResolvedHighlight
) -> list[etree._Element]:
    x, y, width, _height = bbox
    overlays: list[etree._Element] = []

    if highlight.symbol:
        glyph = SYMBOL_GLYPHS.get(highlight.symbol, "?")
        bx, by = x + width, y
        overlays.append(
            _svg_el(
                "circle", cx=bx, cy=by, r=9, fill="#ffffff", stroke="#333333", **{"stroke-width": 1}
            )
        )
        overlays.append(_text_el(glyph, bx, by + 4, "middle", "11px"))

    if highlight.label:
        overlays.append(
            _text_el(
                highlight.label,
                x,
                y - 6,
                "start",
                "10px",
                fill=highlight.color or DEFAULT_TEXT_COLOR,
                weight="bold",
            )
        )

    return overlays


def _gateway_name_caption(
    bbox: tuple[float, float, float, float], label: str
) -> etree._Element:
    """A wrapped caption below the diamond, showing the gateway's real BPMN name."""
    x, y, width, height = bbox
    lines = _wrap_label(label, max(width * 3, 140), font_size=10.0)
    cx, caption_y = x + width / 2, y + height + 14
    return _multiline_text_el(lines, cx, caption_y, "middle", "10px", DEFAULT_TEXT_COLOR)


def _element_id_caption(bbox: tuple[float, float, float, float], element_id: str) -> etree._Element:
    """A small caption above the shape, showing its BPMN element id."""
    x, y, width, _height = bbox
    cx = x + width / 2
    text = _truncate(element_id, max(width + 20, 100), font_size=9.0)
    return _text_el(text, cx, y - 4, "middle", "9px", fill="#666666")


def _flow_condition_label(midpoint: tuple[float, float], label: str) -> etree._Element:
    """A single-line, truncated condition label (e.g. "Yes"/"No") above a flow's midpoint."""
    mx, my = midpoint
    text = _truncate(label, 140, font_size=10.0)
    return _text_el(text, mx, my - 8, "middle", "10px", weight="bold")


def _multiline_text_el(
    lines: list[str], x: float, y: float, anchor: str, font_size: str, fill: str
) -> etree._Element:
    line_height = 12.0
    el = _svg_el(
        "text",
        x=x,
        y=y,
        **{
            "text-anchor": anchor,
            "font-size": font_size,
            "font-family": "sans-serif",
            "fill": fill,
        },
    )
    for i, line in enumerate(lines):
        tspan = _svg_el("tspan", x=x)
        if i > 0:
            tspan.set("dy", str(line_height))
        tspan.text = line
        el.append(tspan)
    return el


def _svg_el(tag: str, **attrs: object) -> etree._Element:
    el = etree.Element(f"{{{SVG_NS}}}{tag}")
    for key, value in attrs.items():
        el.set(key, str(value))
    return el


def _text_el(
    text: str,
    x: float,
    y: float,
    anchor: str,
    font_size: str,
    fill: str = DEFAULT_TEXT_COLOR,
    weight: str | None = None,
) -> etree._Element:
    attrs = {
        "x": x,
        "y": y,
        "text-anchor": anchor,
        "font-size": font_size,
        "font-family": "sans-serif",
        "fill": fill,
    }
    if weight:
        attrs["font-weight"] = weight
    el = _svg_el("text", **attrs)
    el.text = text
    return el
