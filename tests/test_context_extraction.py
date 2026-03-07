from __future__ import annotations

from oarp.context import ContextValidator, MaterialContextExtractor
from oarp.models import RunConfig
from oarp.topic_spec import spec_from_dict


def _spec():
    return spec_from_dict(
        {
            "topic_id": "ni",
            "variables": [
                {"name": "thickness_nm", "unit": "nm", "datatype": "float", "normalization": "length"},
                {"name": "temperature_c", "unit": "c", "datatype": "float", "normalization": "temperature"},
            ],
            "plot": {"primary": {"x": "thickness_nm", "y": "temperature_c"}},
        }
    )


def test_material_context_extractor_parses_substrate_orientation_and_pure_ni() -> None:
    extractor = MaterialContextExtractor()
    spec = _spec()
    cfg = RunConfig(run_dir="/tmp/oarp")

    snippet = "For 10 nm Ni on Si(100) substrate, pure Ni films were annealed at 350 C without dopant."
    ctx = extractor.extract_context(snippet, spec, cfg)

    assert ctx["substrate_material"] in {"Si", "Si(100)", "silicon"}
    assert ctx["substrate_orientation"]
    assert ctx["doping_state"] in {"undoped", "na_pure_ni"}
    assert ctx["alloy_state"] in {"pure", "na_pure_ni"}
    assert bool(ctx["pure_ni_evidence"]) is True


def test_context_validator_enforces_pure_ni_exception() -> None:
    validator = ContextValidator()
    spec = _spec()
    cfg = RunConfig(run_dir="/tmp/oarp", require_context_fields=True)

    row = {
        "substrate_material": "Si",
        "substrate_orientation": "(100)",
        "doping_state": "na_pure_ni",
        "alloy_state": "na_pure_ni",
        "doping_composition": "[]",
        "alloy_composition": "[]",
        "pure_ni_evidence": False,
    }
    reasons = validator.validate_context(row, spec, cfg)
    assert "pure_ni_exception_not_supported_by_evidence" in reasons


def test_material_context_extractor_handles_binary_ni_si_as_pure() -> None:
    extractor = MaterialContextExtractor()
    spec = _spec()
    cfg = RunConfig(run_dir="/tmp/oarp")

    snippet = "Binary Ni/Si stacks on Si (111) were annealed without intentional doping or alloying."
    ctx = extractor.extract_context(snippet, spec, cfg)

    assert ctx["substrate_material"] == "Si"
    assert ctx["substrate_orientation"] == "(111)"
    assert ctx["doping_state"] in {"undoped", "na_pure_ni"}
    assert ctx["alloy_state"] in {"pure", "na_pure_ni"}
    assert bool(ctx["pure_ni_evidence"]) is True


def test_material_context_extractor_ignores_si_as_alloy_element_in_silicide_context() -> None:
    extractor = MaterialContextExtractor()
    spec = _spec()
    cfg = RunConfig(run_dir="/tmp/oarp")

    snippet = "The Ni:Si ratio was 1:1.7 in the silicide layer, with a pure Ni top layer on Si(100)."
    ctx = extractor.extract_context(snippet, spec, cfg)

    assert ctx["alloy_state"] != "alloyed"
    assert bool(ctx["pure_ni_evidence"]) is True


def test_material_context_extractor_ignores_citation_like_orientation_tokens() -> None:
    extractor = MaterialContextExtractor()
    spec = _spec()
    cfg = RunConfig(run_dir="/tmp/oarp")

    snippet = "As reported in [19], a 2022 study examined Ni films on silicon substrates."
    ctx = extractor.extract_context(snippet, spec, cfg)

    assert str(ctx.get("substrate_orientation") or "") == ""
