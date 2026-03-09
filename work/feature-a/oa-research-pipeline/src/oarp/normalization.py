from __future__ import annotations

import re
from typing import Tuple

import pandas as pd

UNIT_ALIASES = {
    "°c": "c",
    "oc": "c",
    "degc": "c",
    "celsius": "c",
    "k": "k",
    "kelvin": "k",
    "nm": "nm",
    "nanometer": "nm",
    "nanometers": "nm",
    "um": "um",
    "μm": "um",
    "micrometer": "um",
    "mm": "mm",
    "ev": "ev",
    "mev": "mev",
}

_ANNEAL_ROLE_RE = re.compile(
    r"\b(?:anneal|annealing|post-anneal|after anneal|formed|transformed|phase appears)\b",
    re.IGNORECASE,
)
_ASDEP_ROLE_RE = re.compile(
    r"\b(?:as[-\s]?deposited|deposition|deposited film|initial film|starting film)\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(s|sec|second|seconds|min|minute|minutes|h|hr|hour|hours)\b", re.IGNORECASE)


def canonical_unit(unit: str) -> str:
    key = unit.strip().lower().replace(" ", "")
    return UNIT_ALIASES.get(key, key)


def normalize_value(value: float, raw_unit: str, target_unit: str) -> Tuple[float, str]:
    src = canonical_unit(raw_unit or "")
    dst = canonical_unit(target_unit or "")

    if not dst or src == dst:
        return value, dst or src

    # Temperature conversions
    if src == "k" and dst == "c":
        return value - 273.15, dst
    if src == "c" and dst == "k":
        return value + 273.15, dst

    # Length conversions
    if src == "um" and dst == "nm":
        return value * 1000.0, dst
    if src == "mm" and dst == "nm":
        return value * 1_000_000.0, dst
    if src == "nm" and dst == "um":
        return value / 1000.0, dst

    # Energy conversions
    if src == "mev" and dst == "ev":
        return value / 1000.0, dst
    if src == "ev" and dst == "mev":
        return value * 1000.0, dst

    # Unknown conversion: keep original
    return value, src or dst


def _infer_thickness_role(variable_name: str, snippet: str) -> tuple[str, str]:
    name = str(variable_name or "").strip().lower()
    text = str(snippet or "")
    if "thickness_asdep_nm" == name or "as_deposited" in name:
        return "asdep", "explicit"
    if "thickness_anneal_nm" == name or "post_anneal" in name:
        return "anneal", "explicit"
    if name != "thickness_nm":
        return "", "explicit"
    if _ANNEAL_ROLE_RE.search(text):
        return "anneal", "inferred"
    if _ASDEP_ROLE_RE.search(text):
        return "asdep", "inferred"
    return "asdep", "legacy_alias"


def _infer_anneal_time_seconds(snippet: str) -> float | None:
    match = _TIME_RE.search(str(snippet or ""))
    if not match:
        return None
    try:
        value = float(match.group(1))
    except Exception:
        return None
    unit = str(match.group(2) or "").lower()
    if unit.startswith("h"):
        return value * 3600.0
    if unit.startswith("min"):
        return value * 60.0
    return value


def build_thickness_views(points_df: pd.DataFrame, *, compat_alias: bool = True) -> pd.DataFrame:
    out = points_df.copy()
    if out.empty:
        for col in (
            "thickness_asdep_nm",
            "thickness_anneal_nm",
            "thickness_role_source",
            "thickness_relation_confidence",
            "anneal_temperature_c",
            "anneal_time_s",
            "thickness_nm",
        ):
            if col not in out.columns:
                out[col] = None if col != "thickness_role_source" else ""
        return out

    if "snippet" not in out.columns:
        out["snippet"] = ""
    if "variable_name" not in out.columns:
        out["variable_name"] = ""
    if "normalized_value" not in out.columns:
        out["normalized_value"] = 0.0

    asdep_vals: list[float | None] = []
    anneal_vals: list[float | None] = []
    role_src: list[str] = []
    anneal_temp_vals: list[float | None] = []
    anneal_time_vals: list[float | None] = []

    for row in out.to_dict(orient="records"):
        var_name = str(row.get("variable_name") or "").strip()
        snippet = str(row.get("snippet") or "")
        try:
            value = float(row.get("normalized_value"))
        except Exception:
            value = 0.0

        role, source = _infer_thickness_role(var_name, snippet)
        asdep = None
        anneal = None
        if role == "asdep":
            asdep = float(value)
        elif role == "anneal":
            anneal = float(value)
        role_src.append(source)
        asdep_vals.append(asdep)
        anneal_vals.append(anneal)

        name = var_name.lower()
        if name in {"temperature_c", "anneal_temperature_c", "annealing_temperature_c"}:
            anneal_temp_vals.append(float(value))
        else:
            anneal_temp_vals.append(None)

        if name in {"anneal_time_s", "annealing_time_s"}:
            anneal_time_vals.append(float(value))
        else:
            anneal_time_vals.append(_infer_anneal_time_seconds(snippet))

    out["thickness_asdep_nm"] = pd.to_numeric(pd.Series(asdep_vals, index=out.index), errors="coerce")
    out["thickness_anneal_nm"] = pd.to_numeric(pd.Series(anneal_vals, index=out.index), errors="coerce")
    out["thickness_role_source"] = role_src
    out["thickness_relation_confidence"] = out["thickness_role_source"].map(
        lambda src: 0.95 if str(src) == "explicit" else (0.75 if str(src) == "inferred" else 0.6)
    )
    out["anneal_temperature_c"] = pd.to_numeric(pd.Series(anneal_temp_vals, index=out.index), errors="coerce")
    out["anneal_time_s"] = pd.to_numeric(pd.Series(anneal_time_vals, index=out.index), errors="coerce")

    if compat_alias:
        if "thickness_nm" not in out.columns:
            out["thickness_nm"] = pd.Series([float("nan")] * len(out), index=out.index, dtype="float64")
        out["thickness_nm"] = pd.to_numeric(out["thickness_nm"], errors="coerce")
        out["thickness_nm"] = out["thickness_asdep_nm"].combine_first(out["thickness_anneal_nm"])

    return out
