"""Public library API — the contract this project exists to provide.

Client code (a web backend, a notebook, a batch job, the CLI in ``cli.py``)
imports from here and nowhere else in the package. ``parse_diagram`` and
``render`` never touch the filesystem or network: they work entirely on
in-memory objects, so the library is embeddable anywhere.

    from bpmn_highlighter import parse_diagram, render

    diagram = parse_diagram(bpmn_source)   # path, XML str/bytes, file-like, or lxml tree
    svg = render(diagram, {"highlights": [...]})  # plain dict, same shape as the YAML/JSON config

Internal modules (``parser``, ``highlight``, ``renderer``) can be refactored
freely as long as this contract holds.
"""

from __future__ import annotations

from typing import Any, Literal

from .highlight import (
    HighlightConfig,
    HighlightConfigError,
    highlight_elements,
    load_highlight_config,
)
from .model import Diagram
from .parser import BpmnSource, parse_diagram
from .renderer import render_svg

RenderBackend = Literal["custom", "pm4py"]

__all__ = [
    "parse_diagram",
    "render",
    "Diagram",
    "HighlightConfig",
    "HighlightConfigError",
    "BpmnSource",
    "RenderBackend",
]

def render(
    diagram: Diagram | BpmnSource,
    highlights: dict[str, Any] | HighlightConfig,
    backend: RenderBackend = "custom",
    show_ids: bool = False,
) -> str:
    """Render ``diagram`` to an SVG string with the given highlight rules applied.

    ``highlights`` may be either:

    - a plain ``dict`` in the highlight-config shape documented in
      CLAUDE.md's "Highlight config format" (the same shape whether it came
      from parsed YAML, parsed JSON, or was built up in code) — it is
      validated internally against ``diagram`` via
      :func:`~bpmn_highlighter.highlight.load_highlight_config`; or
    - an already-validated :class:`HighlightConfig` — e.g. from a caller
      that renders the same highlight set against many diagrams and wants
      to validate it once and reuse it. In this case it is trusted as-is
      and not re-validated against this particular ``diagram``.

    ``backend`` picks which renderer draws the SVG:

    - ``"custom"`` (default) — ``renderer.py``'s hand-drawn SVG, laid out
      from the diagram's own DI bounds/waypoints.
    - ``"pm4py"`` — ``renderer_pm4py.py``, which outsources drawing to
      pm4py's classic (Graphviz) BPMN visualizer instead and overlays
      highlights on top on a best-effort basis; see that module's docstring
      for its id-mapping approach and known gaps (no pool/lane highlighting,
      flow highlights are color/border only). Requires the optional
      ``pm4py`` extra and the Graphviz ``dot`` binary on ``PATH`` — neither
      is imported unless this backend is actually used.

    ``show_ids``, when true, shows each task's element ``id`` alongside it --
    drawn above the rectangle for the ``"custom"`` backend, or appended to the
    task's own label for the ``"pm4py"`` backend (which cannot caption
    gateways/events this way; see ``renderer_pm4py.render_svg``).
    """
    diagram = diagram if isinstance(diagram, Diagram) else parse_diagram(diagram)

    if isinstance(highlights, HighlightConfig):
        config = highlights
    elif isinstance(highlights, dict):
        config = load_highlight_config(highlights, diagram=diagram)
    else:
        raise TypeError(
            f"highlights must be a dict or a HighlightConfig, got {type(highlights).__name__}"
        )

    resolved = highlight_elements(diagram, config)

    if backend == "custom":
        return render_svg(diagram, resolved, show_ids=show_ids)
    if backend == "pm4py":
        from .renderer_pm4py import render_svg as render_svg_pm4py

        return render_svg_pm4py(diagram, resolved, show_ids=show_ids)
    raise ValueError(f"unknown backend {backend!r} (expected 'custom' or 'pm4py')")


from IPython.display import SVG, display

def render_diff(diag1, diag2, diff, backend='custom', show_output=True, show_ids=False, diag1_out=None, diag2_out=None):
    hlhts1 = { 'highlights': [
        * [ { 'match': { 'id': n }, 'color': '#ff0000', 'symbol': 'cross' } for n in diff['nodes']['del'] ],
        * [ { 'match': { 'id': n[2] }, 'color': '#ff0000', 'symbol': 'cross' } for n in diff['edges']['del'] ],
    ] }
    svg1 = render(diag1, hlhts1, backend, show_ids=show_ids)
    if show_output:
        display(SVG(svg1))

    if diag1_out is not None:
        with open(diag1_out, "w", encoding="utf-8") as file:
            file.write(svg1)

    hlhts2 = { 'highlights': [
        * [ { 'match': { 'id': n }, 'color': '#00ff00', 'symbol': 'add' } for n in diff['nodes']['add'] ],
        * [ { 'match': { 'id': n[2] }, 'color': '#00ff00', 'symbol': 'add' } for n in diff['edges']['add'] ],
    ] }
    svg2 = render(diag2, hlhts2, backend, show_ids=show_ids)
    if show_output:
        display(SVG(svg2))

    if diag2_out is not None:
        with open(diag2_out, "w", encoding="utf-8") as file:
            file.write(svg2)