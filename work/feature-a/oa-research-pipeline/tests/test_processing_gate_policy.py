from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from oarp.models import RunConfig
from oarp.processing import evaluate_processor_with_policy
from oarp.runtime import ensure_run_layout


def test_evaluate_processor_with_ceiling_policy_allows_non_degradation(tmp_path: Path, monkeypatch) -> None:
    run_dir = (tmp_path / "run").resolve()
    artifacts = ensure_run_layout(run_dir)["artifacts"]
    model_base = artifacts / "models" / "tabular_head"
    model_ft = artifacts / "models" / "finetune_nisi_sub200"
    model_base.mkdir(parents=True, exist_ok=True)
    model_ft.mkdir(parents=True, exist_ok=True)
    (model_base / "processor_model.pkl").write_bytes(b"x")
    (model_ft / "processor_model.pkl").write_bytes(b"x")

    frame = pd.DataFrame(
        [
            {"article_key": "a1", "phase_label": "NiSi", "thickness_nm": 100.0},
            {"article_key": "a2", "phase_label": "NiSi", "thickness_nm": 120.0},
            {"article_key": "a3", "phase_label": "NiSi", "thickness_nm": 150.0},
        ]
    )
    frame.to_parquet(artifacts / "processor_training_rows.parquet", index=False)
    frame.to_parquet(artifacts / "finetune_system_eval.parquet", index=False)

    def _fake_eval(*, frame: pd.DataFrame, model_path: Path, target_phase: str):  # noqa: ANN001, ARG001
        token = str(model_path)
        if "finetune_nisi_sub200" in token:
            return {"phase_macro_f1": 0.98, "ndcg_at_10": 0.98}
        return {"phase_macro_f1": 0.98, "ndcg_at_10": 0.98}

    monkeypatch.setattr("oarp.processing._evaluate_model_on_frame", _fake_eval)

    cfg = RunConfig(
        run_dir=run_dir,
        finetune_gate_policy="ceiling_aware",
        finetune_ceiling_threshold=0.95,
        finetune_min_slice_rows=2,
        finetune_min_support_articles=2,
        finetune_target_phase="NiSi",
        finetune_max_thickness_nm=200.0,
    )
    payload = evaluate_processor_with_policy(run_dir=run_dir, cfg=cfg)
    assert payload["gate_policy_branch"] == "ceiling_non_degradation"
    assert payload["all_gates_pass"] is True
    assert payload["gates"]["finetune_f1_policy_gate"] is True
    assert payload["gates"]["finetune_ndcg_policy_gate"] is True


def test_evaluate_processor_with_ceiling_policy_mixed_metric_ceiling(tmp_path: Path, monkeypatch) -> None:
    run_dir = (tmp_path / "run_mixed").resolve()
    artifacts = ensure_run_layout(run_dir)["artifacts"]
    model_base = artifacts / "models" / "tabular_head"
    model_ft = artifacts / "models" / "finetune_nisi_sub200"
    model_base.mkdir(parents=True, exist_ok=True)
    model_ft.mkdir(parents=True, exist_ok=True)
    (model_base / "processor_model.pkl").write_bytes(b"x")
    (model_ft / "processor_model.pkl").write_bytes(b"x")

    frame = pd.DataFrame(
        [
            {"article_key": "a1", "phase_label": "NiSi", "thickness_nm": 100.0},
            {"article_key": "a2", "phase_label": "NiSi", "thickness_nm": 120.0},
            {"article_key": "a3", "phase_label": "NiSi", "thickness_nm": 150.0},
            {"article_key": "a4", "phase_label": "NiSi", "thickness_nm": 180.0},
        ]
    )
    frame.to_parquet(artifacts / "processor_training_rows.parquet", index=False)
    frame.to_parquet(artifacts / "finetune_system_eval.parquet", index=False)

    call_idx = {"value": 0}

    def _fake_eval(*, frame: pd.DataFrame, model_path: Path, target_phase: str):  # noqa: ANN001, ARG001
        token = str(model_path)
        if "finetune_nisi_sub200" in token:
            return {"phase_macro_f1": 1.0, "ndcg_at_10": 1.0}
        # evaluate_processor_with_policy invokes base model twice:
        # first on full base frame (base gate), then on eval slice (policy baseline).
        call_idx["value"] += 1
        if call_idx["value"] == 1:
            return {"phase_macro_f1": 0.90, "ndcg_at_10": 1.0}
        return {"phase_macro_f1": 0.45, "ndcg_at_10": 1.0}

    monkeypatch.setattr("oarp.processing._evaluate_model_on_frame", _fake_eval)

    cfg = RunConfig(
        run_dir=run_dir,
        finetune_gate_policy="ceiling_aware",
        finetune_ceiling_threshold=0.95,
        finetune_min_slice_rows=2,
        finetune_min_support_articles=2,
        finetune_target_phase="NiSi",
        finetune_max_thickness_nm=200.0,
    )
    payload = evaluate_processor_with_policy(run_dir=run_dir, cfg=cfg)
    assert payload["gate_policy_branch"] == "ceiling_mixed"
    assert payload["all_gates_pass"] is True
    assert payload["gates"]["finetune_f1_policy_gate"] is True
    assert payload["gates"]["finetune_ndcg_policy_gate"] is True


def test_evaluate_processor_skip_policy_relaxes_low_support_and_resolves_target_from_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = (tmp_path / "run_skip").resolve()
    artifacts = ensure_run_layout(run_dir)["artifacts"]
    model_base = artifacts / "models" / "tabular_head"
    model_ft = artifacts / "models" / "finetune_nisi_sub200"
    model_base.mkdir(parents=True, exist_ok=True)
    model_ft.mkdir(parents=True, exist_ok=True)
    (model_base / "processor_model.pkl").write_bytes(b"x")
    (model_ft / "processor_model.pkl").write_bytes(b"x")
    (model_ft / "processor_model_meta.json").write_text(
        json.dumps({"finetune_target_phase": "NiSi"}, indent=2),
        encoding="utf-8",
    )

    frame = pd.DataFrame(
        [
            {"article_key": "a1", "phase_label": "NiSi", "thickness_nm": 80.0},
            {"article_key": "a1", "phase_label": "NiSi", "thickness_nm": 90.0},
            {"article_key": "a2", "phase_label": "NiSi", "thickness_nm": 100.0},
        ]
    )
    frame.to_parquet(artifacts / "processor_training_rows.parquet", index=False)
    frame.to_parquet(artifacts / "finetune_system_eval.parquet", index=False)

    observed_targets: list[str] = []

    def _fake_eval(*, frame: pd.DataFrame, model_path: Path, target_phase: str):  # noqa: ANN001, ARG001
        observed_targets.append(str(target_phase))
        return {"phase_macro_f1": 0.99, "ndcg_at_10": 0.99}

    monkeypatch.setattr("oarp.processing._evaluate_model_on_frame", _fake_eval)

    cfg = RunConfig(
        run_dir=run_dir,
        finetune_gate_policy="ceiling_aware",
        finetune_ceiling_threshold=0.95,
        finetune_min_slice_rows=20,
        finetune_min_support_articles=5,
        finetune_target_phase="",
        finetune_max_thickness_nm=200.0,
        finetune_on_insufficient="skip",
    )
    payload = evaluate_processor_with_policy(run_dir=run_dir, cfg=cfg)

    assert payload["target_phase"] == "NiSi"
    assert payload["finetune_execution_status"] == "skipped_insufficient_support"
    assert payload["gates"]["finetune_support_rows_gate"] is True
    assert payload["gates"]["finetune_support_articles_gate"] is True
    assert payload["all_gates_pass"] is True
    assert "NiSi" in observed_targets
