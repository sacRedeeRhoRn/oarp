from __future__ import annotations

from oarp.discovery import RollingSaturationController


def test_saturation_controller_triggers_on_low_recent_yield() -> None:
    ctrl = RollingSaturationController(window_pages=3, min_yield=0.05, min_pages_before=4)
    metrics = [
        {"page_yield": 0.20},
        {"page_yield": 0.12},
        {"page_yield": 0.03},
        {"page_yield": 0.01},
        {"page_yield": 0.00},
    ]
    assert ctrl.should_stop("k", metrics) is True


def test_saturation_controller_waits_for_min_pages() -> None:
    ctrl = RollingSaturationController(window_pages=3, min_yield=0.05, min_pages_before=6)
    metrics = [
        {"page_yield": 0.01},
        {"page_yield": 0.00},
        {"page_yield": 0.00},
    ]
    assert ctrl.should_stop("k", metrics) is False
