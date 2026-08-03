"""Date-stamped ``outputs/`` layout.

Every run writes flat into ``outputs/<DDMonYYYY>/`` (today's date, no
subfolders) -- e.g. ``outputs/28Jul2026/model_checkpoint.pt``. Reading a bare
filename (no directory component) searches today's folder first, then falls
back to the most recent earlier date folder that has it, so a pipeline stage
(e.g. ``contest_all.py`` looking for ``model_checkpoint.pt``) can find
yesterday's (or last week's) output without the caller re-specifying a full
path every day. A path that already has a directory component is an explicit
override and is always returned unchanged.
"""

import os
from datetime import date, datetime
from pathlib import Path

OUTPUTS_ROOT = Path("outputs")
DATE_FORMAT = "%d%b%Y"  # e.g. 28Jul2026


def today_output_dir() -> Path:
    """Today's date folder under ``outputs/``, created if missing."""
    d = OUTPUTS_ROOT / date.today().strftime(DATE_FORMAT)
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_path(filename: str) -> str:
    """Path to write ``filename`` to -- always today's date folder."""
    return str(today_output_dir() / filename)


def _date_dirs_newest_first() -> list[Path]:
    if not OUTPUTS_ROOT.exists():
        return []
    dated = []
    for p in OUTPUTS_ROOT.iterdir():
        if not p.is_dir():
            continue
        try:
            parsed = datetime.strptime(p.name, DATE_FORMAT).date()
        except ValueError:
            continue
        dated.append((parsed, p))
    dated.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in dated]


def find_output(filename: str) -> str:
    """Path to read ``filename`` from: today's date folder if it's there,
    else the most recent earlier date folder that has it. Falls back to
    today's (possibly nonexistent) path if no date folder has the file, so
    error messages still point somewhere sensible."""
    for d in _date_dirs_newest_first():
        candidate = d / filename
        if candidate.exists():
            return str(candidate)
    return str(today_output_dir() / filename)


def resolve_read_path(name_or_path: str) -> str:
    """For paths being *read*: a bare filename is looked up via
    ``find_output`` (today's folder, else the most recent earlier date
    folder that has it); a path with a directory component (an explicit
    override) is returned unchanged."""
    if os.path.dirname(name_or_path):
        return name_or_path
    return find_output(name_or_path)


def resolve_write_path(name_or_path: str) -> str:
    """For paths being *written*: a bare filename always goes to today's
    date folder; a path with a directory component (an explicit override)
    is returned unchanged."""
    if os.path.dirname(name_or_path):
        return name_or_path
    return output_path(name_or_path)
