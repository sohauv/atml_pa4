# ATML PA4 reproduction

Run commands from the repository root. GPU experiments use CUDA and the pinned Transformers release in the course data requirements. For local summaries and plots, use Python 3.12 with NumPy and Matplotlib.

## Data and environment

```bash
git clone https://github.com/AbDu11aHHH/atml-pa4-datasets.git
git -C atml-pa4-datasets checkout 6a0a9ec9da7ef3d313ee3036cfec493886723644
python -m pip install -r atml-pa4-datasets/requirements-public.txt
python -m pip install -r requirements-analysis.txt
cp source_revisions.json atml-pa4-datasets/data/local_source_revisions.json
python atml-pa4-datasets/scripts/setup_data.py --public
python atml-pa4-datasets/scripts/check_setup.py
```

Install the CUDA build of PyTorch for your GPU environment before running GPU experiments. The saved run metadata records the actual PyTorch, CUDA, and Transformers versions where available. The course data repository supplies frozen prompt selections and the MQAR generator. HellaSwag formatting is adapted from lm-evaluation-harness v0.4.9.

## Task 1

```bash
python baseline.py --course-data atml-pa4-datasets --output baseline-rerun.json
python pruning.py --course-data atml-pa4-datasets --output-dir runs/pruning
python pruning.py --course-data atml-pa4-datasets --output-dir runs/q1 --conditions uniform_int8 uniform_int4 uniform_int3
python pruning.py --course-data atml-pa4-datasets --output-dir runs/awq --conditions folded_int8 folded_int4 folded_int3 folded_mixed_int4_budget folded_mixed_int3_budget
```

The folded runs are the final Q2/Q3 comparisons. Packed model sizes are estimates and the timing runs use dense FP16 execution. The folded mixed-precision allocator ranks candidates using pre-fold calibration reconstruction errors.

## Task 2

```bash
python attention_benchmark.py --output runs/attention/attention.json
python attention_benchmark.py --causality-only --output runs/attention/causality.json
python mqar.py --course-data atml-pa4-datasets --output-dir runs/mqar
python mqar.py --course-data atml-pa4-datasets --output-dir runs/intervention --methods curriculum --inference-sweep
python kv_cache.py --course-data atml-pa4-datasets --output-dir runs/kv --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
python kv_cache.py --course-data atml-pa4-datasets --output-dir runs/kv-reference --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --reference-only
```

The token-wise, channel-wise, and hybrid cache layouts preserve an incomplete append group in FP16. The corrected retrieval runs combine the original FP16 context residual with appended positions before forming complete groups. The append-only boundary check is recorded separately. The benchmark uses chunked prefill to construct the cache for larger batch sizes.

## Task 3

```bash
python speculative.py --markov-output markov-rerun.json
python distillation.py --course-data atml-pa4-datasets --output-dir runs/distillation
python speculative_eval.py --course-data atml-pa4-datasets --adapters runs/distillation --output-dir runs/speculative
```

Both GPU runners can continue from an existing output directory. Evaluation recreates each merged draft from its trained adapter and pinned backbone. The final teacher and original-draft alignment scores use banks produced by the course setup script. The Transformers compatibility adjustment applies the course per-prompt seed to CUDA before generation. The original sequentially generated banks remain in the initial evaluation archive. The saved timing traces record per-round costs and adjacent target-only runs.

## Targeted GPU checks

```bash
python awq_fold_check.py
python kv_boundary_check.py
python padding_check.py
```

These checks write JSON in separate subdirectories under `runs/`.

## Rebuild the summaries and figures

```bash
python summarize_task1.py
python summarize_task2.py
python summarize_speculative.py pa4-speculative-results.zip task3-summary.json --training-archive pa4-distillation-results.zip --course-bank-archive pa4-course-bank-results.zip
python plot_results.py
python select_examples.py
```

The result archives contain per-prompt or per-condition JSON used by these scripts. The submission staging copies omit binary training checkpoints. To regenerate adapters and checkpoints, run the training commands above.

Course data and MQAR generator: https://github.com/AbDu11aHHH/atml-pa4-datasets

HellaSwag preprocessing: https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9/lm_eval/tasks/hellaswag/utils.py
