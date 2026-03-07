from __future__ import annotations

from typing import Tuple

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
