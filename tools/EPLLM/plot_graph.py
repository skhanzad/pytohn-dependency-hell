#!/usr/bin/env python3
"""
Plot the compiled EPLLM LangGraph and save it as a PNG.

Usage:
    python plot_graph.py [output.png]

If no output path is given, saves to graph.png in the current directory.
"""

import sys
from pathlib import Path

# Ensure we can import from graph.py in the same directory
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from graph import build_graph


def main() -> None:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _here / "graph.png"

    graph = build_graph()
    drawable = graph.get_graph()

    try:
        png_bytes = drawable.draw_mermaid_png()
    except Exception as e:
        print(
            "draw_mermaid_png() failed (you may need: pip install grandalf). "
            "Falling back to Mermaid source.",
            file=sys.stderr,
        )
        print(str(e), file=sys.stderr)
        mermaid = drawable.draw_mermaid()
        out_mmd = output_path.with_suffix(".mmd")
        out_mmd.write_text(mermaid, encoding="utf-8")
        print(f"Mermaid diagram written to {out_mmd}", file=sys.stderr)
        sys.exit(1)

    output_path.write_bytes(png_bytes)
    print(f"Graph saved to {output_path}")


if __name__ == "__main__":
    main()
