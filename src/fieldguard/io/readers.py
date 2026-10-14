"""Read CSV/XLSX survey exports as all-text frames (keeps IDs like '00123' intact)."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd

from fieldguard.config import DataConfig

_NA_VALUES = ["", "NA", "N/A", "n/a", "NaN", "nan", "NULL", "null", "None", "#N/A", "."]


def sniff_delimiter(sample: str) -> str:
    """Guess the delimiter from the header line (Kobo exports default to ';')."""
    header = sample.splitlines()[0] if sample else ""
    counts = {d: header.count(d) for d in (",", ";", "\t", "|")}
    best = max(counts, key=lambda d: counts[d])
    if counts[best] > 0:
        return best
    try:
        return csv.Sniffer().sniff(sample[:4096]).delimiter
    except csv.Error:
        return ","


def read_csv_text(path: str | Path, delimiter: str | None = None, encoding: str = "utf-8-sig") -> pd.DataFrame:
    p = Path(path)
    with p.open("r", encoding=encoding, errors="replace", newline="") as fh:
        sample = fh.read(65536)
    sep = delimiter or sniff_delimiter(sample)
    return pd.read_csv(
        p,
        sep=sep,
        dtype=str,
        encoding=encoding,
        encoding_errors="replace",
        keep_default_na=False,
        na_values=_NA_VALUES,
        low_memory=False,
    )


def read_excel_text(path: str | Path, sheet: str | int | None = 0) -> pd.DataFrame:
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ImportError("Reading .xlsx needs the 'excel' extra: pip install 'fieldguard[excel]'") from exc
    df = pd.read_excel(path, sheet_name=0 if sheet is None else sheet, dtype=str, engine="openpyxl")
    return _strip_blank(df)


def read_table(
    path: str | Path, sheet: str | int | None = 0, delimiter: str | None = None, encoding: str = "utf-8-sig"
) -> pd.DataFrame:
    """Read a CSV/TSV/XLSX export into a frame of strings (missing values are NaN)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"data file not found: {p}")
    suffix = p.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        df = read_excel_text(p, sheet)
    elif suffix in {".csv", ".tsv", ".txt"}:
        df = read_csv_text(
            p, delimiter="\t" if suffix == ".tsv" and delimiter is None else delimiter, encoding=encoding
        )
    elif suffix == ".parquet":
        raw = pd.read_parquet(p)
        df = _strip_blank(raw.astype("str").mask(raw.isna()))
    else:
        raise ValueError(f"unsupported file type {suffix!r}: use .csv, .tsv, .xlsx or .parquet")
    df.columns = [str(c).strip() for c in df.columns]
    return df.reset_index(drop=True)


def read_from_config(cfg: DataConfig, base_dir: Path) -> pd.DataFrame:
    if not cfg.path:
        raise ValueError("data.path is not set (or use a connector via `fieldguard fetch`)")
    p = Path(cfg.path)
    if not p.is_absolute():
        p = base_dir / p
    return read_table(p, sheet=cfg.sheet, delimiter=cfg.delimiter, encoding=cfg.encoding)


def frame_from_csv_string(text: str) -> pd.DataFrame:
    """Convenience for tests and notebooks."""
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, na_values=_NA_VALUES)


def _strip_blank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
            s = s.astype("str").where(s.notna())
            s = s.str.strip()
            out[col] = s.mask(s.isin(_NA_VALUES))
    return out
