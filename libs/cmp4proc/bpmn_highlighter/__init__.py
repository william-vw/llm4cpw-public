"""bpmn_highlighter — render BPMN 2.0 diagrams with config-driven highlighting.

Public API re-exported from :mod:`bpmn_highlighter.api`; see that module's
docstring for the full contract.
"""

from .api import (
    BpmnSource,
    Diagram,
    HighlightConfig,
    HighlightConfigError,
    RenderBackend,
    parse_diagram,
    render,
    render_diff,
)

__all__ = [
    "parse_diagram",
    "render",
    "render_diff",
    "Diagram",
    "HighlightConfig",
    "HighlightConfigError",
    "BpmnSource",
    "RenderBackend",
]

__version__ = "0.1.0"
