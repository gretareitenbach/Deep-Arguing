"""Mirror substantive (non-progress) terminal output into a markdown file under
``outputs/logs/``, so results/metrics/diagnostics survive after the terminal
scrollback is gone -- console/logging output is unaffected, this just also
writes the same content to disk.

Lines of the form ``"--- X ---"`` become level-2 headings; a line already
containing a newline (e.g. a pre-rendered code block) is written through
as-is; everything else becomes a bullet point.
"""

from pathlib import Path


def write_markdown_log(lines: list[str], path: str, mode: str = "a") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rendered = []
    for line in lines:
        if "\n" in line:
            rendered.append(line)
            continue
        stripped = line.strip()
        if stripped.startswith("---") and stripped.endswith("---"):
            rendered.append(f"## {stripped.strip('- ').strip()}")
        else:
            rendered.append(f"- {stripped}")

    with open(path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(f"# {Path(path).stem} log\n\n")
        f.write("\n".join(rendered) + "\n\n")
