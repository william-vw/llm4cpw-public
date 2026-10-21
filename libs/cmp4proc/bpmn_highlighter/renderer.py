"""Diagram model + resolved highlights -> SVG.

Draws the base diagram (shapes, flows, labels) from the DI geometry captured
by ``parser.py``, then layers highlight fill/border/symbol/label styling on
top. Returns SVG as a plain ``str`` — the caller decides whether to write it
to disk, stream it in an HTTP response, or rasterize it elsewhere. This
module never touches the filesystem.

Kept as small, composable per-element-type draw functions (``_draw_task``,
``_draw_gateway``, ``_draw_event``, ...) rather than one large branching
function, per CLAUDE.md conventions.
"""

from __future__ import annotations

import math

import svgwrite

from .highlight import ResolvedHighlight
from .model import Bounds, Diagram, DiagramElement
from .parser import DATA_TYPES, EVENT_TYPES, GATEWAY_TYPES, TASK_TYPES

MARGIN = 20.0

DEFAULT_FILL = "#ffffff"
DEFAULT_STROKE = "#333333"
DEFAULT_TEXT_COLOR = "#1a1a1a"
CONTAINER_TYPES = {"pool", "lane"}

# Named border styles a highlight rule's "border" field may use; anything
# else is passed straight through as a raw SVG stroke-dasharray value, so
# e.g. border: "6,2,2,2" also works.
BORDER_DASH_STYLES = {
    "solid": None,
    "dashed": "8,4",
    "dotted": "2,3",
}

# Small glyph shown in a badge at the shape's top-right corner. Keep in sync
# with highlight.BUILTIN_SYMBOLS.
SYMBOL_GLYPHS = {
    "warning": "⚠",
    "robot": "\U0001f916",
    "clock": "⏰",
    "star": "★",
    "check": "✓",
    "cross": "✗",
    "flag": "⚑",
    "lock": "\U0001f512",
    "info": "ℹ",
    "add": "➕"
}


def render_svg(
    diagram: Diagram, resolved: dict[str, ResolvedHighlight], show_ids: bool = False
) -> str:
    """Render ``diagram`` to an SVG string, applying ``resolved`` highlights.

    ``show_ids``, when true, draws each task's element ``id`` in small text
    above its rectangle -- useful when authoring/debugging highlight configs
    that match on ``id``.
    """
    shapes = diagram.shapes()
    edges = diagram.edges()

    dwg = _new_drawing(shapes, edges)

    # Draw pools/lanes first (as background containers), then flows, then
    # flow nodes on top, so nothing meaningful is visually buried.
    containers = [el for el in shapes if el.type in CONTAINER_TYPES]
    flow_nodes = [el for el in shapes if el.type not in CONTAINER_TYPES]
    created_markers: set[str] = set()

    # Base layer -- every element's plain shape, no highlight overlays yet.
    # Highlighted flow nodes are drawn after the plain ones so an adjacent
    # unhighlighted shape can't paint over a highlighted one.
    for el in containers:
        _draw_element(dwg, el, resolved.get(el.id))
    flow_midpoints: dict[str, tuple[float, float] | None] = {}
    for el in edges:
        flow_midpoints[el.id] = _draw_flow(dwg, el, resolved.get(el.id), created_markers)
    plain_nodes = [el for el in flow_nodes if resolved.get(el.id) is None]
    highlighted_nodes = [el for el in flow_nodes if resolved.get(el.id) is not None]
    for el in [*plain_nodes, *highlighted_nodes]:
        _draw_element(dwg, el, resolved.get(el.id))

    # Highlight layer -- symbol badges and highlight labels, drawn last so
    # no later shape or flow can obscure them.
    for el in [*containers, *flow_nodes]:
        _draw_overlays(dwg, el, resolved.get(el.id))
    for el in edges:
        midpoint = flow_midpoints.get(el.id)
        if midpoint is not None:
            _draw_flow_overlays(dwg, resolved.get(el.id), midpoint)

    if show_ids:
        for el in flow_nodes:
            if el.type in TASK_TYPES:
                _draw_element_id(dwg, el)

    return str(dwg.tostring())


def _new_drawing(shapes: list[DiagramElement], edges: list[DiagramElement]) -> svgwrite.Drawing:
    xs: list[float] = []
    ys: list[float] = []
    for el in shapes:
        b = el.bounds
        assert b is not None
        xs.extend([b.x, b.x + b.width])
        ys.extend([b.y, b.y + b.height])
    for el in edges:
        xs.extend(p.x for p in el.waypoints)
        ys.extend(p.y for p in el.waypoints)

    if not xs or not ys:
        return svgwrite.Drawing(size=("200px", "100px"), profile="full", viewBox="0 0 200 100")

    min_x, max_x = min(xs) - MARGIN, max(xs) + MARGIN
    min_y, max_y = min(ys) - MARGIN, max(ys) + MARGIN
    width, height = max_x - min_x, max_y - min_y

    return svgwrite.Drawing(
        size=(f"{width:.0f}px", f"{height:.0f}px"),
        profile="full",
        viewBox=f"{min_x:.0f} {min_y:.0f} {width:.0f} {height:.0f}",
    )


def _dash_attr(dasharray: str | None) -> dict[str, str]:
    # svgwrite's attribute validator chokes on an explicit None value (it
    # tries to iterate it), so an absent dasharray must be an absent kwarg,
    # not stroke_dasharray=None.
    return {"stroke_dasharray": dasharray} if dasharray else {}


def _style(highlight: ResolvedHighlight | None) -> tuple[str, str, str | None]:
    """Return (fill, stroke, dasharray) for an element given its highlight."""
    if highlight is None:
        return DEFAULT_FILL, DEFAULT_STROKE, None
    fill = highlight.color or DEFAULT_FILL
    stroke = highlight.color or DEFAULT_STROKE
    dasharray = None
    if highlight.border:
        dasharray = BORDER_DASH_STYLES.get(highlight.border, highlight.border)
    return fill, stroke, dasharray


def _draw_element(
    dwg: svgwrite.Drawing, element: DiagramElement, highlight: ResolvedHighlight | None
) -> None:
    if element.bounds is None:
        return

    if element.type in TASK_TYPES:
        _draw_task(dwg, element, highlight)
    elif element.type in EVENT_TYPES:
        _draw_event(dwg, element, highlight)
    elif element.type in GATEWAY_TYPES:
        _draw_gateway(dwg, element, highlight)
    elif element.type in CONTAINER_TYPES:
        _draw_pool_lane(dwg, element, highlight)
    elif element.type in DATA_TYPES:
        _draw_data(dwg, element, highlight)
    else:
        _draw_unknown(dwg, element, highlight)


def _draw_task(
    dwg: svgwrite.Drawing, element: DiagramElement, highlight: ResolvedHighlight | None
) -> None:
    b = element.bounds
    assert b is not None
    fill, stroke, dasharray = _style(highlight)
    dwg.add(
        dwg.rect(
            insert=(b.x, b.y),
            size=(b.width, b.height),
            rx=6,
            ry=6,
            fill=fill,
            stroke=stroke,
            stroke_width=2,
            **_dash_attr(dasharray),
        )
    )
    _draw_center_label(dwg, element, b)


def _draw_gateway(
    dwg: svgwrite.Drawing, element: DiagramElement, highlight: ResolvedHighlight | None
) -> None:
    b = element.bounds
    assert b is not None
    fill, stroke, dasharray = _style(highlight)
    cx, cy = b.x + b.width / 2, b.y + b.height / 2
    points = [(cx, b.y), (b.x + b.width, cy), (cx, b.y + b.height), (b.x, cy)]
    dwg.add(
        dwg.polygon(
            points=points,
            fill=fill,
            stroke=stroke,
            stroke_width=2,
            **_dash_attr(dasharray),
        )
    )
    _draw_gateway_marker(dwg, element, b)
    _draw_center_label(dwg, element, b, dy=b.height / 2 + 14)


def _draw_gateway_marker(dwg: svgwrite.Drawing, element: DiagramElement, b: Bounds) -> None:
    """Draw the inner glyph: an "X" for an exclusive gateway, a "+" otherwise."""
    cx, cy = b.x + b.width / 2, b.y + b.height / 2
    r = min(b.width, b.height) / 4
    if element.type == "exclusiveGateway":
        # Shrink the diagonals by 1/sqrt(2) so the "X" reads the same visual
        # size as the "+" rather than filling a larger bounding box.
        d = r / math.sqrt(2)
        segments = [(-d, -d, d, d), (-d, d, d, -d)]
    else:
        segments = [(-r, 0.0, r, 0.0), (0.0, -r, 0.0, r)]
    for dx1, dy1, dx2, dy2 in segments:
        dwg.add(
            dwg.line(
                start=(cx + dx1, cy + dy1),
                end=(cx + dx2, cy + dy2),
                stroke=DEFAULT_STROKE,
                stroke_width=3,
                stroke_linecap="round",
            )
        )


def _draw_event(
    dwg: svgwrite.Drawing, element: DiagramElement, highlight: ResolvedHighlight | None
) -> None:
    b = element.bounds
    assert b is not None
    fill, stroke, dasharray = _style(highlight)
    cx, cy = b.x + b.width / 2, b.y + b.height / 2
    r = min(b.width, b.height) / 2
    stroke_width = 3 if element.type == "endEvent" else 2
    dwg.add(
        dwg.circle(
            center=(cx, cy),
            r=r,
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
            **_dash_attr(dasharray),
        )
    )
    _draw_center_label(dwg, element, b, dy=b.height / 2 + 14)


def _draw_pool_lane(
    dwg: svgwrite.Drawing, element: DiagramElement, highlight: ResolvedHighlight | None
) -> None:
    b = element.bounds
    assert b is not None
    fill = highlight.color if highlight and highlight.color else "none"
    _, stroke, dasharray = _style(highlight)
    dwg.add(
        dwg.rect(
            insert=(b.x, b.y),
            size=(b.width, b.height),
            fill=fill,
            fill_opacity=0.15 if fill != "none" else 1,
            stroke=stroke,
            stroke_width=2,
            **_dash_attr(dasharray),
        )
    )
    if element.label:
        dwg.add(
            dwg.text(
                _truncate(element.label, b.width - 12),
                insert=(b.x + 6, b.y + 14),
                fill=DEFAULT_TEXT_COLOR,
                font_size="11px",
                font_family="sans-serif",
                font_weight="bold",
            )
        )


def _draw_data(
    dwg: svgwrite.Drawing, element: DiagramElement, highlight: ResolvedHighlight | None
) -> None:
    b = element.bounds
    assert b is not None
    fill, stroke, dasharray = _style(highlight)
    dwg.add(
        dwg.rect(
            insert=(b.x, b.y),
            size=(b.width, b.height),
            fill=fill,
            stroke=stroke,
            stroke_width=2,
            **_dash_attr(dasharray),
        )
    )
    _draw_center_label(dwg, element, b, dy=b.height / 2 + 12)


def _draw_unknown(
    dwg: svgwrite.Drawing, element: DiagramElement, highlight: ResolvedHighlight | None
) -> None:
    b = element.bounds
    assert b is not None
    fill, stroke, dasharray = _style(highlight)
    dwg.add(
        dwg.rect(
            insert=(b.x, b.y),
            size=(b.width, b.height),
            fill=fill,
            stroke=stroke,
            stroke_width=1.5,
            stroke_dasharray=dasharray or "4,2",
        )
    )
    _draw_center_label(dwg, element, b)


def _draw_flow(
    dwg: svgwrite.Drawing,
    element: DiagramElement,
    highlight: ResolvedHighlight | None,
    created_markers: set[str],
) -> tuple[float, float] | None:
    """Draw the flow's polyline and return its midpoint (for the overlay pass)."""
    if len(element.waypoints) < 2:
        return None
    stroke = (highlight.color if highlight else None) or DEFAULT_STROKE
    stroke_width = 2.5 if highlight else 1.5
    dasharray = None
    if highlight and highlight.border:
        dasharray = BORDER_DASH_STYLES.get(highlight.border, highlight.border)
    points = [(p.x, p.y) for p in element.waypoints]
    marker_id = _ensure_arrowhead_marker(dwg, stroke, created_markers)
    dwg.add(
        dwg.polyline(
            points=points,
            fill="none",
            stroke=stroke,
            stroke_width=stroke_width,
            **_dash_attr(dasharray),
            marker_end=f"url(#{marker_id})",
        )
    )
    return _midpoint(points)


def _ensure_arrowhead_marker(dwg: svgwrite.Drawing, color: str, created_markers: set[str]) -> str:
    # One marker per distinct arrow color, since <marker> ids are global to
    # the document and each flow's arrowhead must match its own stroke color.
    marker_id = f"arrowhead-{color.lstrip('#')}"
    if marker_id not in created_markers:
        marker = dwg.marker(
            insert=(8, 3), size=(10, 6), orient="auto", id=marker_id, markerUnits="strokeWidth"
        )
        marker.add(dwg.path(d="M0,0 L10,3 L0,6 Z", fill=color))
        dwg.defs.add(marker)
        created_markers.add(marker_id)
    return marker_id


# Rough advance width of one character relative to the font size, for a
# sans-serif face -- good enough to keep labels from overflowing their shape.
_CHAR_WIDTH_EM = 0.58


def _truncate(text: str, max_width: float, font_size: float = 11.0) -> str:
    """Shorten ``text`` with an ellipsis so it fits within ``max_width`` px."""
    if max_width <= 0:
        return text
    max_chars = max(1, int(max_width / (font_size * _CHAR_WIDTH_EM)))
    if len(text) <= max_chars:
        return text
    if max_chars == 1:
        return text[0]
    return text[: max_chars - 1].rstrip() + "…"


def _wrap_label(
    text: str, max_width: float, max_lines: int = 3, font_size: float = 11.0
) -> list[str]:
    """Greedy word-wrap ``text`` into at most ``max_lines`` lines.

    Each returned line fits within ``max_width`` px; if the text still runs
    past the last allowed line, that line is ellipsized via :func:`_truncate`.
    """
    if max_width <= 0:
        return [text]
    max_chars = max(1, int(max_width / (font_size * _CHAR_WIDTH_EM)))
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = ""
    for i, word in enumerate(words):
        candidate = f"{current} {word}".strip()
        if not current or len(candidate) <= max_chars:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) == max_lines - 1:
            # No more lines available -- dump the rest onto this one and let
            # _truncate() below clip it with an ellipsis.
            current = " ".join(words[i:])
            break
    lines.append(current)
    return [_truncate(line, max_width, font_size) for line in lines]


def _draw_center_label(
    dwg: svgwrite.Drawing,
    element: DiagramElement,
    b: Bounds,
    dy: float = 0.0,
    max_width: float | None = None,
) -> None:
    if not element.label:
        return
    # Labels drawn inside the shape (dy == 0) must fit its width; labels
    # placed below a small shape (gateways, events) get a looser allowance so
    # they stay readable.
    if max_width is not None:
        limit = max_width
    elif dy == 0:
        limit = b.width - 8
    else:
        limit = max(b.width * 3, 140)

    lines = _wrap_label(element.label, limit)
    if not lines:
        return

    line_height = 12.0
    cx = b.x + b.width / 2
    base_y = b.y + b.height / 2 + dy
    # Inside a shape (dy == 0) the wrapped block is vertically centered; when
    # the label sits below the shape (dy != 0) the given y is the first
    # baseline and extra lines grow downward.
    start_y = base_y - (len(lines) - 1) * line_height / 2 if dy == 0 else base_y

    text_el = dwg.text(
        "",
        insert=(cx, start_y),
        fill=DEFAULT_TEXT_COLOR,
        font_size="11px",
        font_family="sans-serif",
        text_anchor="middle",
    )
    for i, line in enumerate(lines):
        text_el.add(dwg.tspan(line, x=[cx], dy=[0 if i == 0 else line_height]))
    dwg.add(text_el)


def _draw_overlays(
    dwg: svgwrite.Drawing, element: DiagramElement, highlight: ResolvedHighlight | None
) -> None:
    if highlight is None:
        return
    b = element.bounds
    if b is None:
        return

    if highlight.symbol:
        glyph = SYMBOL_GLYPHS.get(highlight.symbol, "?")
        bx, by = b.x + b.width, b.y
        dwg.add(dwg.circle(center=(bx, by), r=9, fill="#ffffff", stroke="#333333", stroke_width=1))
        dwg.add(
            dwg.text(
                glyph,
                insert=(bx, by + 4),
                text_anchor="middle",
                font_size="11px",
                font_family="sans-serif",
            )
        )

    if highlight.label:
        dwg.add(
            dwg.text(
                _truncate(highlight.label, max(b.width * 2, 120), font_size=10.0),
                insert=(b.x, b.y - 6),
                fill=highlight.color or DEFAULT_TEXT_COLOR,
                font_size="10px",
                font_family="sans-serif",
                font_weight="bold",
            )
        )


def _draw_element_id(dwg: svgwrite.Drawing, element: DiagramElement) -> None:
    """Draw ``element.id`` in small text centered above its rectangle."""
    b = element.bounds
    if b is None:
        return
    dwg.add(
        dwg.text(
            _truncate(element.id, b.width + 20, font_size=9.0),
            insert=(b.x + b.width / 2, b.y - 4),
            text_anchor="middle",
            fill="#666666",
            font_size="9px",
            font_family="monospace",
        )
    )


def _midpoint(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Point at 50% of the polyline's total length (not just its bounding box)."""
    if len(points) == 1:
        return points[0]

    # Adjacent-pairs sliding window: points and points[1:] differ in length
    # by design, so this pairing is intentionally non-`strict`.
    segments = list(zip(points, points[1:], strict=False))
    lengths = [math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in segments]
    total = sum(lengths)
    if total == 0:
        return points[0]

    target = total / 2
    covered = 0.0
    for ((x1, y1), (x2, y2)), seg_len in zip(segments, lengths, strict=True):
        if covered + seg_len >= target:
            t = (target - covered) / seg_len if seg_len else 0.0
            return (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
        covered += seg_len
    return points[-1]


def _draw_flow_overlays(
    dwg: svgwrite.Drawing, highlight: ResolvedHighlight | None, midpoint: tuple[float, float]
) -> None:
    if highlight is None:
        return
    mx, my = midpoint

    if highlight.symbol:
        glyph = SYMBOL_GLYPHS.get(highlight.symbol, "?")
        by = my - 14
        dwg.add(dwg.circle(center=(mx, by), r=9, fill="#ffffff", stroke="#333333", stroke_width=1))
        dwg.add(
            dwg.text(
                glyph,
                insert=(mx, by + 4),
                text_anchor="middle",
                font_size="11px",
                font_family="sans-serif",
            )
        )

    if highlight.label:
        label_y = my - 28 if highlight.symbol else my - 10
        dwg.add(
            dwg.text(
                highlight.label,
                insert=(mx, label_y),
                text_anchor="middle",
                fill=highlight.color or DEFAULT_TEXT_COLOR,
                font_size="10px",
                font_family="sans-serif",
                font_weight="bold",
            )
        )