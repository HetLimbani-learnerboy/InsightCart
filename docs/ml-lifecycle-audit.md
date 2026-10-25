# InsightCart ML lifecycle audit

## Existing architecture

- Dataset ingestion and required-column checks live in `src/components/`.
- `DataTransformation` removes incomplete and duplicate rows, makes a stratified
  60/10/10/20 train, validation, promotion, and test split, and fits TF-IDF,
  category encoding, and rating scaling on training rows only.
- `ModelTrainer` owns the candidate estimator registry and selects by validation
  F1 with ROC-AUC as the tie-breaker. `ModelEvaluation` reports the final test
  metrics and records a training run in MLflow.
- `run_v2_experiment.py` evaluates candidate and production against the same
  promotion holdout. `ModelPusher` is called only when `--promote` is explicit
  and the quality gate passes. FastAPI loads the active production bundle or
  the legacy model and preprocessing files.
- DVC runs the candidate pipeline. GitHub Actions has separate validation and
  training jobs; Docker Compose, Prometheus, and API tracing are configured in
  their existing files.

## Confirmed problems and risks

- CI looked for `data/processed/model_comparison.csv`, while training wrote
  `artifacts/experiments/<candidate>/model_comparison.csv`. The upload also
  warned rather than failed when required results were missing.
- `ModelTrainerConfig` had a stale comparison CSV default outside the experiment
  directory.
- The recorded candidate's promotion F1 was `0.8971316`; production scored
  `0.9544780` on the identical promotion holdout. The gate's default minimum
  improvement is zero and it still requires a strict improvement over
  production. This is why the gate rejected the candidate. Validation and final
  test F1 are not promotion inputs.
- The known production model checksum is
  `5d9257f3ed19e1a2a8fcd39ab14ed251993a5ae08417c1e8472358109956e157`; it
  remained unchanged during this work.
- The active shell has `DEBUG=release`; app tests need `DEBUG=false` in this
  environment. DVC graph inspection could not open its local database. The
  installed scikit-learn version also differs from the version used to serialize
  production artifacts, producing compatibility warnings during local
  evaluation.
- The repository has API/contract/tracing and model-selection tests, but no
  dedicated CI-output validation test, output-schema test, or recovery test for
  interrupted multi-file promotion. Cloud DVC/S3 access and Docker deployment
  were not verified here.

## Changes made

- `src/components/model_trainer.py`: correct default comparison path and include
  the requested algorithm, parameters, training status, error, and validation
  metric columns while retaining existing report columns.
- `src/pipeline/run_v2_experiment.py`: add a machine-readable promotion decision
  report and record the experiment timestamp/ID, Git SHA, MLflow run ID, and
  candidate output checksums in metadata.
- `.github/workflows/ml-pipeline.yml`: train into the DVC candidate directory,
  fail on absent or invalid mandatory outputs, and upload only after successful
  validation.
- `README.md`: document the actual command, paths, and the observed reason for
  the gate rejection.

## Validation sequence and outstanding work

1. Run lint and the existing test suite.
2. Train a candidate into an isolated temporary directory and validate the
   generated artifacts and metadata.
3. Check the production model checksum.
4. Run DVC and remote S3 validation in an environment with a writable DVC cache
   and configured AWS credentials; run the GitHub Actions workflow remotely.
5. Add promotion interruption/recovery and end-to-end API compatibility tests
   before changing production promotion behavior further.
