"""CLI — argument parsing and file I/O only.

No parsing/matching/rendering logic lives here; every command delegates to
``api.parse_diagram()`` / ``api.render()`` exactly like any other library
caller. If a fix only works "from the CLI", that's a sign logic leaked in
here and needs to move into ``api.py`` or the internal modules instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import yaml

from . import api


@click.group()
@click.version_option(package_name="bpmn-highlighter")
def main() -> None:
    """Render BPMN 2.0 diagrams with config-driven element highlighting."""


@main.command()
@click.argument("bpmn_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--highlights",
    "highlights_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="YAML or JSON file describing which elements to highlight.",
)
@click.option(
    "-o",
    "--output",
    "output_file",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the rendered SVG.",
)
@click.option(
    "--backend",
    type=click.Choice(["custom", "pm4py"]),
    default="custom",
    show_default=True,
    help="'pm4py' outsources drawing to pm4py's Graphviz-based BPMN visualizer "
    "instead of the built-in renderer (requires the 'pm4py' extra and Graphviz).",
)
def render(
    bpmn_file: Path, highlights_file: Path, output_file: Path, backend: api.RenderBackend
) -> None:
    """Render BPMN_FILE to SVG with the highlight rules from --highlights applied."""
    diagram = api.parse_diagram(bpmn_file)
    highlights = _load_highlights_file(highlights_file)

    try:
        svg = api.render(diagram, highlights, backend=backend)
    except api.HighlightConfigError as exc:
        raise click.ClickException(f"invalid highlight config: {exc}") from exc

    output_file.write_text(svg, encoding="utf-8")
    click.echo(f"Wrote {output_file}")


def _load_highlights_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        raise click.ClickException(
            f"Unsupported highlights file extension {path.suffix!r} "
            "(expected .yaml, .yml, or .json)"
        )
    if not isinstance(data, dict):
        raise click.ClickException(f"{path} must contain a mapping/object at the top level")
    return data


if __name__ == "__main__":
    main()
