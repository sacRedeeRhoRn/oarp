# OA Research Pipeline (`oarp`)

Topic-agnostic open-access research extraction package that can:
- discover OA research papers from public APIs,
- acquire full text with policy-aware crawling,
- extract text/table/figure evidence as normalized points,
- validate and consensus-rank points with uncertainty,
- generate citation-backed plots and reports.

New in v0.3+:
- global thin-film knowledge artifacts (`phase_events`, `condition_graph`, `quality_outcomes`)
- predictive-first recipe generation with progressive gates (`loose -> balanced -> strict -> best`)
- async jobs API/service for non-interactive orchestration.

## Requirements
- Python `>=3.11`
- Optional: Docker for reproducible heavy-parser profiles.
- v0.6 strict processor path:
  - `torch` + `torch_geometric` (for `processor-train-gnn`)
  - `xgboost` or `lightgbm` recommended (tabular head), sklearn fallback remains available

System-level prerequisites (outside venv):
- `python3.11`
- Docker daemon
- compiler toolchain for wheel builds

Venv-level setup handled by strict bootstrap:
- create run-local `.venv`
- install `oa-research-pipeline[ml,pdf,html]`
- verify strict imports and write dependency snapshots

## Install

```bash
python -m pip install -e .
```

Strict bootstrap from near-clean machine:

```bash
python3.11 scripts/bootstrap_strict.py --run /tmp/oarp_strict --profile strict_full --local-repo-path /Users/moon.s.june/matGen/MODELING/collect/pdf_repo
```

## CLI

```bash
oarp run --spec examples/topic_ni_silicide.yaml --query "Depending on thickness of as-deposited Nickel film, at which temperature do Ni2Si, NiSi, NiSi2 appear?" --out ./runs/ni --plugin materials_ni_silicide --min-discovery-score 0.62 --max-pages-per-provider 60 --max-discovered-records 100000 --saturation-window-pages 8 --saturation-min-yield 0.03 --min-pages-before-saturation 20 --resume-discovery --max-downloads 2000 --acquire-workers 8 --extract-workers 4 --english-first --per-provider-cap arxiv:3000,openalex:3000 --require-fulltext-mime --context-window-lines 2 --point-assembler sentence-window --context-assembler sentence-window --require-context-fields
```

Pipeline subcommands:

```bash
oarp discover --spec <topic.yaml> --query "<question>" --out <run_dir> --min-discovery-score 0.62 --max-pages-per-provider 60 --max-discovered-records 100000 --saturation-window-pages 8 --saturation-min-yield 0.03 --min-pages-before-saturation 20 --resume-discovery --plugin <plugin_id> --emit-debug-ranking
oarp acquire --run <run_dir> --max-downloads 2000 --acquire-workers 8 --english-first --resume-discovery --require-fulltext-mime
oarp extract --run <run_dir> --engines text,table,figure --extract-workers 4 --context-window-lines 2 --point-assembler sentence-window --context-assembler sentence-window --require-context-fields --plugin <plugin_id>
oarp validate --run <run_dir> --require-context-fields --emit-validation-metrics
oarp consensus --run <run_dir>
oarp plot --run <run_dir>
oarp report --run <run_dir>
oarp benchmark --spec <topic.yaml> --gold <gold_dir> --gold-context examples/gold_ni_context --context-completeness-gate 0.70 --context-precision-gate 0.80 --out <bench_dir> --shadow-gold examples/gold_ni_shadow --strict-gold
oarp agents-inc-handoff --run <run_dir> --out <handoff.json>
oarp mp-enrich --run <run_dir> --mp-mode interpreter --mp-scope summary_thermo --mp-on-demand --mp-query-workers 4 --mp-timeout-sec 20 --mp-max-queries 20000 --mp-cache-path <run_dir>/artifacts/materials_project_cache.sqlite

# New in v0.5: parallel SLM swarm extraction + calibration
oarp extract --run <run_dir> --extractor-mode slm_swarm --extractor-models llama-3.1-8b-instruct,gemma-2-9b-it,phi-3-mini-4k-instruct --schema-decoder llguidance --vote-policy weighted --min-vote-confidence 0.60
oarp validate --run <run_dir> --emit-extraction-calibration --calibration-bins 10

# New in v0.6: mandatory TGI SLM extraction mode
oarp extract --run <run_dir> --extractor-mode slm_tgi_required --tgi-endpoint http://localhost:8080/generate --tgi-models llama-3.1-8b-instruct,gemma-2-9b-it --tgi-workers 4 --decoder jsonschema --slm-max-retries 2 --slm-timeout-sec 20 --slm-batch-size 8 --slm-chunk-tokens 384 --slm-overlap-tokens 96 --slm-eval-split all
oarp validate --run <run_dir> --extractor-gate-profile strict_v1 --emit-extraction-calibration --calibration-bins 10

# New in v0.7.1: enhanced bootstrap + all-done validation pack
oarp bootstrap-strict --out <run_dir> --python-exec <run_dir>/.venv/bin/python --tgi-platform auto --tgi-mode docker
oarp validate-all-done --spec <topic.yaml> --query "<question>" --out <run_dir> --gold <gold_dir> --shadow-gold <shadow_gold_dir> --gold-context <gold_context_dir> --precision-gate 0.80 --context-completeness-gate 0.70 --context-precision-gate 0.80

# New in v0.5: graph+tabular processor train/fine-tune + benchmark namespaces
oarp process-train-universal --run <run_dir> --datasets phase_events --out <model_dir>
oarp process-finetune-system --run <run_dir> --system-id nickel-silicide --base-model <model_dir> --out <finetuned_model_dir>
oarp benchmark-extraction --suite thinfilm_gold --spec <topic.yaml> --run <run_dir> --gold <gold_dir> --out <bench_dir>
oarp benchmark-processing --suite phase_transition --run <run_dir> --out <bench_dir>

# New in v0.6: explicit processor phase commands (GNN -> tabular -> finetune -> eval)
oarp processor-train-gnn --run <run_dir> --epochs 80 --batch-size 32 --lr 1e-3 --hidden-dim 64 --layers 2 --dropout 0.1
oarp processor-train-tabular --run <run_dir> --base-gnn <gnn_model_dir> --model xgboost
oarp processor-finetune --run <run_dir> --base-model <tabular_model_dir> --target NiSi --max-thickness-nm 200 --epochs 40
oarp processor-eval --run <run_dir> --suite base_and_finetune

# New: knowledge + recipes
oarp knowledge --run <run_dir>
oarp recipe-generate --run <run_dir> --request-file examples/recipe_request_ni.json --mp-enabled --mp-api-key "$MP_API_KEY" --mp-mode interpreter --mp-scope summary_thermo
oarp recipe-generate --run <run_dir> --request-file examples/recipe_request_ni.json --processor-model <finetuned_model_dir> --gate-profile progressive_default

# New: async service + jobs
oarp service start --data-dir ~/.oarp_service --host 127.0.0.1 --port 8787 --mp-enabled --mp-api-key "$MP_API_KEY"
oarp job submit-research --spec examples/topic_ni_silicide.yaml --query "nickel silicide thin film phase transition"
oarp job submit-recipe --request-file examples/recipe_request_ni.json
oarp job status --job-id <job_id>
oarp job artifacts --job-id <job_id>
oarp job retry --job-id <job_id>
oarp job run-worker --max-jobs 20 --mp-enabled --mp-api-key "$MP_API_KEY"
```

## Python API

```python
from oarp.pipeline import RunConfig, run_pipeline
from oarp.service_api import submit_recipe_generation_job, submit_research_index_job, get_job_status
from oarp.service_models import RecipeGenerateRequest, ServiceConfig

result = run_pipeline(
    spec_path="examples/topic_ni_silicide.yaml",
    query="Depending on thickness of as-deposited Nickel film, at which temperature do Ni2Si, NiSi, NiSi2 appear?",
    cfg=RunConfig(run_dir="./runs/ni")
)
print(result.report_path)

svc = ServiceConfig(data_dir="~/.oarp_service")
research_job = submit_research_index_job("examples/topic_ni_silicide.yaml", svc)
print(research_job.job_id, research_job.status)

recipe_req = RecipeGenerateRequest(
    target_film_material="NiSi",
    target_phase_target={"type": "phase_label", "value": "NiSi"},
    knowledge_run_dir="./runs/ni",
    text_constraints={"method_family": ["PVD", "ALD"]},
)
recipe_job = submit_recipe_generation_job(recipe_req, svc)
print(get_job_status(recipe_job.job_id, svc))
```

## Data Artifacts
Generated under `run_dir/artifacts/`:
- `articles.parquet`
- `articles_ranked.parquet`
- `discovery_debug.parquet`
- `discovery_pages.parquet`
- `discovery_cursors.json`
- `documents.parquet`
- `crawl_progress.parquet`
- `acquisition_debug.parquet`
- `document_quality.parquet`
- `evidence_points.parquet`
- `assembled_points.parquet`
- `extraction_coverage.parquet`
- `context_dimensions.parquet`
- `extraction_votes.parquet`
- `extraction_error_slices.parquet`
- `extraction_calibration.parquet`
- `slm_requests.parquet`
- `slm_responses.parquet`
- `slm_points_raw.parquet`
- `slm_points_voted.parquet`
- `slm_eval_metrics.json`
- `phase_events.parquet`
- `condition_graph.parquet`
- `quality_outcomes.parquet`
- `materials_project_enriched_points.parquet`
- `materials_project_materials.parquet`
- `materials_project_point_links.parquet`
- `materials_project_query_log.parquet`
- `materials_project_conflicts.parquet`
- `materials_project_cache.sqlite`
- `recipe_candidates.parquet`
- `recipe_ranked.parquet`
- `recipe_cards.json`
- `processor_training_rows.parquet`
- `graph_samples.parquet`
- `gnn_train_metrics.json`
- `tabular_train_metrics.json`
- `processor_eval_base.json`
- `processor_eval_finetune.json`
- `finetune_slice.parquet`
- `processor_eval_metrics.json`
- `universal_model_card.md`
- `system_finetune_card.md`
- `gate_audit.parquet`
- `safety_audit.parquet`
- `materials_project_refs.json`
- `provenance.parquet`
- `validated_points.parquet`
- `rejected_points.parquet`
- `validation_reasons.parquet`
- `context_validation.parquet`
- `consensus_points.parquet`
- `run_metrics.json`
- `stage_metrics.parquet`
- `bootstrap_report.json`
- `preflight_report.json`
- `full_workflow_metrics.json`
- `full_workflow_stage_metrics.parquet`
- `all_done_validation.json`
- `all_done_validation.md`
- `repro_compare.json`
- `dependency_lock_snapshot.txt`
- `cpu_env_probe.json`
- `index.sqlite`
- service jobs metadata: `jobs.sqlite`
- benchmark outputs (`benchmark.json`, `benchmark.md`, optional `benchmark_shadow.json`, `benchmark_shadow.md`, `benchmark_context.json`, `benchmark_context.md`, `benchmark_mp.json`, `benchmark_mp.md`)
- plots and `report.md`

## Notes
- Discovery starts with OpenAlex/Crossref/arXiv/Europe PMC.
- Acquisition is API-first with OA direct crawl fallback.
- Figure extraction is plugin-based and can be extended with external engines.
- Materials Project enrichment is optional (set `MP_API_KEY` or pass `--mp-api-key`).
