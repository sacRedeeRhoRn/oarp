from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from oarp.models import MaterialContext, RunConfig
from oarp.topic_spec import TopicSpec

_ORIENTATION_RE = re.compile(
    r"(\(\s*-?\d\s*-?\d\s*-?\d\s*\)|\[\s*-?\d\s*-?\d\s*-?\d\s*\]|<\s*-?\d\s*-?\d\s*-?\d\s*>)"
)
_COMPOSITION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(at\.?%|wt\.?%|mol\.?%|%)\s*([A-Z][a-z]?)",
    re.IGNORECASE,
)
_DOPED_RE = re.compile(r"\b([A-Z][a-z]?)\s*[- ]?doped\b|\bdoped with\s+([A-Z][a-z]?)", re.IGNORECASE)
_ALLOY_RE = re.compile(r"\bni\s*\(\s*([A-Z][a-z]?)\s*\)|\bni\s*[- ]\s*([A-Z][a-z]?)\b", re.IGNORECASE)

_SUBSTRATE_PATTERNS = [
    re.compile(r"\bon\s+([a-z0-9\-\(\)\[\]<> ]{2,50})\s+substrate\b", re.IGNORECASE),
    re.compile(r"\b([a-z0-9\-\(\)\[\]<> ]{2,50})\s+substrate\b", re.IGNORECASE),
]

_SUBSTRATE_CANONICAL = {
    "silicon": "Si",
    "si": "Si",
    "si wafer": "Si",
    "si(100)": "Si",
    "si(111)": "Si",
    "sio2": "SiO2",
    "silica": "SiO2",
    "quartz": "SiO2",
    "sapphire": "Sapphire",
    "al2o3": "Al2O3",
    "gaas": "GaAs",
    "ge": "Ge",
    "glass": "Glass",
}

_PURE_NI_PATTERNS = [
    re.compile(r"\bpure\s+ni\b", re.IGNORECASE),
    re.compile(r"\bpure\s+nickel\b", re.IGNORECASE),
    re.compile(r"\bundoped\s+ni\b", re.IGNORECASE),
    re.compile(r"\bundoped\s+nickel\b", re.IGNORECASE),
    re.compile(r"\bnominally\s+undoped\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+intentional\s+doping\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+intentional\s+alloy(?:ing|ed)?\b", re.IGNORECASE),
    re.compile(r"\bno\s+dopant\b", re.IGNORECASE),
    re.compile(r"\bno\s+alloy(?:ing|ed)?\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+dop(?:ing|ant)\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+alloy(?:ing|ed)?\b", re.IGNORECASE),
    re.compile(r"\bunalloyed\b", re.IGNORECASE),
    re.compile(r"\bbinary\s+ni\s*/\s*si\b", re.IGNORECASE),
    re.compile(r"\bni\s*/\s*(?:n-?si|p-?si|si)\b", re.IGNORECASE),
    re.compile(r"\b(?:as[- ]deposited\s+)?ni(?:ckel)?\s+(?:thin\s+)?(?:film|layer|interlayer)s?\b", re.IGNORECASE),
]

_ELEMENT_SYMBOLS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
    "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
    "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
}

_SILICIDE_CONTEXT_RE = re.compile(r"\bsilicide\b|\bni\s*[:/]\s*si\b|\bni\s*2\s*si\b|\bnisi2?\b", re.IGNORECASE)
_UNDOPED_RE = re.compile(
    r"\bundoped\b|\bno\s+dop(?:ant|ing)\b|\bwithout\s+(?:intentional\s+)?dop(?:ant|ing)\b",
    re.IGNORECASE,
)
_PURE_ALLOY_RE = re.compile(
    r"\bunalloyed\b|\bpure\s+ni(?:ckel)?\b|\bwithout\s+(?:intentional\s+)?alloy(?:ing|ed)?\b|\bno\s+alloy(?:ing|ed)?\b",
    re.IGNORECASE,
)


def _is_element_symbol(token: str) -> bool:
    clean = str(token or "").strip().capitalize()
    return clean in _ELEMENT_SYMBOLS


def _canonical_substrate(raw: str) -> str:
    text = re.sub(r"\s+", " ", raw.strip().lower())
    if not text:
        return ""
    if text in _SUBSTRATE_CANONICAL:
        return _SUBSTRATE_CANONICAL[text]
    # Partial-key fallback for phrases like "single crystal silicon"
    for key, val in _SUBSTRATE_CANONICAL.items():
        if key in text:
            return val
    return raw.strip()


def _orientation_family(value: str) -> str:
    text = re.sub(r"\s+", "", value.strip())
    if not text:
        return ""
    if text.lower() in {"c-plane", "a-plane", "m-plane"}:
        return text.lower()
    digits = re.sub(r"[^0-9]", "", text)
    if not digits:
        return ""
    return f"<{digits}>"


def _extract_substrate(snippet: str) -> str:
    lower = snippet.lower()
    for pattern in _SUBSTRATE_PATTERNS:
        match = pattern.search(lower)
        if not match:
            continue
        val = str(match.group(1) or "").strip()
        if val:
            return _canonical_substrate(val)
    # Direct dictionary fallback
    for key, val in _SUBSTRATE_CANONICAL.items():
        if re.search(rf"\b{re.escape(key)}\b", lower):
            return val
    return ""


def _extract_orientation(snippet: str) -> str:
    lower = snippet.lower()
    for item in ["c-plane", "a-plane", "m-plane"]:
        if item in lower:
            return item
    match = _ORIENTATION_RE.search(snippet)
    if match:
        return re.sub(r"\s+", "", str(match.group(1) or "").strip())
    return ""


def _extract_compositions(snippet: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in _COMPOSITION_RE.finditer(snippet):
        try:
            value = float(match.group(1))
        except Exception:
            continue
        basis = str(match.group(2) or "").strip().lower().replace(".", "")
        element = str(match.group(3) or "").strip().capitalize()
        if not element:
            continue
        if not _is_element_symbol(element):
            continue
        rows.append(
            {
                "element": element,
                "value": value,
                "basis": basis,
            }
        )
    return rows


def _extract_doping(snippet: str) -> tuple[str, list[str], list[dict[str, Any]]]:
    elements: list[str] = []
    for match in _DOPED_RE.finditer(snippet):
        e1 = str(match.group(1) or "").strip()
        e2 = str(match.group(2) or "").strip()
        token = (e1 or e2).capitalize()
        if token and _is_element_symbol(token) and token not in elements:
            elements.append(token)

    comps = [item for item in _extract_compositions(snippet) if item.get("element") in elements or elements == []]

    if elements:
        return "doped", elements, comps
    if _UNDOPED_RE.search(snippet):
        return "undoped", [], []
    return "", [], []


def _extract_alloy(snippet: str) -> tuple[str, list[str], list[dict[str, Any]]]:
    silicide_context = bool(_SILICIDE_CONTEXT_RE.search(snippet))
    elements: list[str] = []
    for match in _ALLOY_RE.finditer(snippet):
        e1 = str(match.group(1) or "").strip()
        e2 = str(match.group(2) or "").strip()
        token = (e1 or e2).capitalize()
        if silicide_context and token.lower() == "si":
            continue
        if token and token.lower() != "ni" and _is_element_symbol(token) and token not in elements:
            elements.append(token)

    if re.search(r"\bni1-x[a-z]{1,2}x\b", snippet, flags=re.IGNORECASE):
        token_match = re.search(r"\bni1-x([a-z]{1,2})x\b", snippet, flags=re.IGNORECASE)
        if token_match:
            token = token_match.group(1).capitalize()
            if silicide_context and token.lower() == "si":
                token = ""
            if _is_element_symbol(token) and token not in elements:
                elements.append(token)

    comps = [item for item in _extract_compositions(snippet) if item.get("element") in elements or elements == []]
    if elements:
        return "alloyed", elements, comps
    if _PURE_ALLOY_RE.search(snippet):
        return "pure", [], []
    # Pure/undoped Ni is modeled through the explicit Day 3 exception state.
    # Keep alloy state empty here so caller can promote to `na_pure_ni`
    # only when direct evidence patterns are present.
    return "", [], []


def _has_ni_matrix_context(snippet: str) -> bool:
    return bool(
        re.search(
            r"\b(?:as[- ]deposited\s+)?ni(?:ckel)?\s+(?:thin\s+)?(?:film|layer|interlayer)s?\b|\bni\s*/\s*(?:n-?si|p-?si|si)\b",
            snippet,
            flags=re.IGNORECASE,
        )
    )


class MaterialContextExtractor:
    def extract_context(self, snippet: str, spec: TopicSpec, cfg: RunConfig) -> dict[str, Any]:  # noqa: ARG002
        text = str(snippet or "")
        substrate = _extract_substrate(text)
        orientation = _extract_orientation(text)
        orientation_family = _orientation_family(orientation)

        doping_state, doping_elements, doping_composition = _extract_doping(text)
        alloy_state, alloy_elements, alloy_composition = _extract_alloy(text)

        pure_ni_evidence = any(pattern.search(text) for pattern in _PURE_NI_PATTERNS)
        if not pure_ni_evidence and _has_ni_matrix_context(text):
            # Treat explicit Ni film/layer stack wording as pure-Ni evidence
            # when no explicit doped/alloyed signal is present.
            if doping_state != "doped" and alloy_state != "alloyed":
                pure_ni_evidence = True
        if pure_ni_evidence:
            if not doping_state:
                doping_state = "na_pure_ni"
            if not alloy_state:
                alloy_state = "na_pure_ni"

        score = 0.0
        if substrate:
            score += 0.3
        if orientation:
            score += 0.25
        if doping_state:
            score += 0.2
        if alloy_state:
            score += 0.2
        if pure_ni_evidence:
            score += 0.05

        context = MaterialContext(
            substrate_material=substrate,
            substrate_orientation=orientation,
            orientation_family=orientation_family,
            doping_state=doping_state,
            doping_elements=doping_elements,
            doping_composition=doping_composition,
            alloy_state=alloy_state,
            alloy_elements=alloy_elements,
            alloy_composition=alloy_composition,
            context_confidence=min(1.0, score),
            pure_ni_evidence=bool(pure_ni_evidence),
        )
        out = asdict(context)
        # Store list-valued payloads as JSON for parquet/sqlite-friendly scalar columns.
        out["doping_elements"] = json.dumps(context.doping_elements, sort_keys=True)
        out["doping_composition"] = json.dumps(context.doping_composition, sort_keys=True)
        out["alloy_elements"] = json.dumps(context.alloy_elements, sort_keys=True)
        out["alloy_composition"] = json.dumps(context.alloy_composition, sort_keys=True)
        return out


class ContextValidator:
    def validate_context(self, point_row: dict, spec: TopicSpec, cfg: RunConfig) -> list[str]:  # noqa: ARG002
        reasons: list[str] = []

        substrate = str(point_row.get("substrate_material") or "").strip()
        orientation = str(point_row.get("substrate_orientation") or "").strip()
        doping_state = str(point_row.get("doping_state") or "").strip().lower()
        alloy_state = str(point_row.get("alloy_state") or "").strip().lower()
        pure_ni_evidence = bool(point_row.get("pure_ni_evidence"))

        if not substrate:
            reasons.append("missing_substrate")
        if not orientation:
            reasons.append("missing_orientation")

        allowed_doping = {"doped", "undoped", "na_pure_ni"}
        if doping_state not in allowed_doping:
            reasons.append("missing_doping_context")

        allowed_alloy = {"alloyed", "pure", "na_pure_ni"}
        if alloy_state not in allowed_alloy:
            reasons.append("missing_alloy_context")

        if (doping_state == "na_pure_ni" or alloy_state == "na_pure_ni") and not pure_ni_evidence:
            reasons.append("pure_ni_exception_not_supported_by_evidence")

        for key in ("doping_composition", "alloy_composition"):
            value = point_row.get(key)
            if value in (None, ""):
                continue
            try:
                parsed = json.loads(value) if isinstance(value, str) else value
            except Exception:
                reasons.append("invalid_composition_format")
                continue
            if not isinstance(parsed, list):
                reasons.append("invalid_composition_format")

        return sorted(set(reasons))
