from __future__ import annotations

import pytest

from oarp.topic_spec import spec_from_dict


def test_topic_spec_validates_minimum_payload() -> None:
    spec = spec_from_dict(
        {
            "topic_id": "demo",
            "keywords": ["k1"],
            "variables": [
                {
                    "name": "thickness_nm",
                    "aliases": ["thickness"],
                    "unit": "nm",
                    "datatype": "float",
                    "normalization": "length",
                    "min_value": 0,
                },
                {
                    "name": "temperature_c",
                    "aliases": ["temperature"],
                    "unit": "c",
                    "datatype": "float",
                    "normalization": "temperature",
                    "min_value": 20,
                    "max_value": 1400,
                },
            ],
            "plot": {"primary": {"x": "thickness_nm", "y": "temperature_c", "color_by": "entity", "transparency_policy": "outlier-transparent"}},
            "plugins": {"preferred_plugin": "materials_ni_silicide"},
        }
    )
    assert spec.topic_id == "demo"
    assert spec.plot.primary.x == "thickness_nm"
    assert spec.plugins.preferred_plugin == "materials_ni_silicide"


def test_topic_spec_rejects_missing_variables() -> None:
    with pytest.raises(ValueError):
        spec_from_dict(
            {
                "topic_id": "bad",
                "plot": {"primary": {"x": "x", "y": "y"}},
                "variables": [],
            }
        )
