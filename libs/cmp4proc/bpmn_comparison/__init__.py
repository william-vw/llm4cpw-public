"""BPMN structural similarity comparison.

A separate, optional top-level package (sibling to ``bpmn_highlighter``,
not part of its public surface — see ``bpmn_highlighter/api.py``) for
comparing two BPMN diagrams' structural similarity. Diagrams are loaded via
``bpmn_highlighter.parser.parse_diagram``, so it accepts the same inputs
(path, XML str/bytes, file-like object, lxml tree) or an already-parsed
``Diagram``. Requires the ``cmp`` extra (``pip install -e ".[cmp]"``).
"""

from .struct_simil import CompareBPMN

__all__ = ["CompareBPMN"]
