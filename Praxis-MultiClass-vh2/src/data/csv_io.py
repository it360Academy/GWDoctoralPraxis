"""Shared CSV reading helper for raw NIS files.

Two real-world HCUP quirks this module works around, both observed on
actual deployment (not hypothetical):

1. **Encoding**: HCUP NIS CSV exports are not always strict UTF-8 - stray
   bytes from Windows-1252/Latin-1 source encoding occasionally show up
   (usually in a rarely-touched text field), which makes pandas' default
   UTF-8 reader raise UnicodeDecodeError partway through the file. Latin-1
   never raises a decode error (every byte 0-255 maps to some valid
   character), so it's the default here.

2. **Delimiter**: despite the ".csv" extension, some HCUP distributions
   are NOT comma-delimited (pipe "|", tab, and semicolon have all been
   observed in the wild for HCUP-family exports). Assuming a comma when
   the real delimiter is something else doesn't raise an error on the
   header line - pandas happily parses a whole header as a tiny handful
   of columns - so it fails SILENTLY at first (a Core file reporting 2-3
   columns instead of 100+) and only raises loudly later, once a data row
   happens to contain the *real* delimiter character in a way that breaks
   comma-based column-count consistency. This module auto-detects the
   delimiter from the first line rather than assuming a comma.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..logging_utils import get_logger

NIS_CSV_ENCODING = "latin1"
CANDIDATE_DELIMITERS = [",", "|", "\t", ";"]
MIN_PLAUSIBLE_COLUMNS = 10  # a real NIS Core/DX_PR_GRPS header has 20-975+ columns


def detect_delimiter(path: Path, encoding: str = NIS_CSV_ENCODING) -> str:
    """Pick whichever candidate delimiter appears most often in the
    file's first line. Falls back to comma if none of the candidates
    appear at all (e.g. a genuinely single-column file)."""
    path = Path(path)
    with open(path, "r", encoding=encoding, errors="replace") as f:
        first_line = f.readline()
    if not first_line:
        return ","
    counts = {d: first_line.count(d) for d in CANDIDATE_DELIMITERS}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def read_nis_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Drop-in replacement for pd.read_csv defaulting to NIS_CSV_ENCODING
    and an auto-detected delimiter. Pass encoding=... or sep=/delimiter=...
    explicitly to override either."""
    log = get_logger("h1.csv_io")
    kwargs.setdefault("encoding", NIS_CSV_ENCODING)

    if "sep" not in kwargs and "delimiter" not in kwargs and args:
        path = Path(args[0])
        if path.exists():
            detected = detect_delimiter(path, encoding=kwargs["encoding"])
            if detected != ",":
                log.warning(
                    "%s does not appear to be comma-delimited despite its "
                    "extension - auto-detected delimiter %r. If this is "
                    "wrong, pass sep=... explicitly to override.",
                    path.name, detected,
                )
            kwargs["sep"] = detected

    df = pd.read_csv(*args, **kwargs)

    if ("chunksize" in kwargs or "iterator" in kwargs):
        return df  # TextFileReader iterator - can't check column count here

    if len(df.columns) < MIN_PLAUSIBLE_COLUMNS and args:
        path = Path(args[0])
        log.warning(
            "%s parsed to only %d column(s) - this is implausibly few for "
            "a real NIS file. The auto-detected delimiter (%r) may be "
            "wrong, or this file may use a format this loader doesn't "
            "handle (e.g. genuinely fixed-width, not delimited at all). "
            "Inspect the file's first line manually before trusting "
            "downstream results.", path.name, len(df.columns), kwargs.get("sep", ","),
        )
    return df

