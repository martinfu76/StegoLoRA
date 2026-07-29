# StegoLoRA Experiment Design

This document separates adapter-learning experiments from watermark-channel
experiments. A LoRA route failure and a watermark decode failure are different
failure modes and should not be reported as one metric.

## Recommended practical profile

The default `practical` profile contains seven staged runs. Unlike a broad
one-factor sweep, it follows the decisions needed to build a deployable
adapter: establish failure modes, add rejection data, compress the adapter,
then reduce data/time or bias training toward safety.

| Run | Change from baseline | Practical question |
| --- | --- | --- |
| `baseline` | all-linear, r=16, no hard negatives | Measure the unprotected trigger behavior |
| `robust_h50` | 50% hard negatives | Establish whether rejection examples fix false activation |
| `robust_qv` | robust training, q_proj/v_proj only | Reduce adapter size while retaining rejection behavior |
| `robust_qv_r8` | robust Q/V adapter with r=8 | Compress rank after module compression |
| `robust_qv_r8_small` | 150 positive + 150 negative | Measure the data-cost floor of the compact candidate |
| `robust_qv_r8_epoch1` | 1 epoch | Measure the training-time floor of the compact candidate |
| `robust_qv_r8_negheavy` | 300 positive + 450 negative, 67% hard | Test a safety-biased operating point |

The baseline uses LR 2e-4, dropout 0.05, and `alpha / r = 2`. Rank experiments
keep `alpha / r` fixed so rank capacity is not confounded with update scale.
This profile omits broad learning-rate, dropout, and alpha sweeps. Run three
seeds only for the final one or two candidates after this staged search.

The existing profiles remain available:

- `smoke`: four quick integration runs.
- `practical`: seven recommended product-oriented comparisons.
- `core`: broader one-factor research sweep.
- `full`: seed repeats, regularization, larger data, and interactions.

## Separate watermark-channel checks

`bits_per_token` and message length affect GPT-2 carrier capacity and fluency,
not the Llama routing adapter. Sweep them with `watermark_sweep.py` rather than
retraining LoRA:

| Axis | Suggested values | Main observation |
| --- | --- | --- |
| Bits per token | 1, 2, 3 | Capacity versus fluency and round-trip stability |
| Message size | 4, 8, 16 characters | Payload length versus recovery and latency |
| MCP samples | 20-50 on the selected adapter | Controller/tool transport reliability |

The activation contract is: only the exact `<|EXTRACT|>` marker at the start of
the user content activates extraction. Training hard negatives cover carriers
without a trigger, multiple near-trigger spellings, quoted markers, and exact
markers embedded later in a normal request. Evaluation uses disjoint prompts,
spellings, and templates; it does not reuse the training attack strings.

## Metrics

Primary routing metrics:

- `correct_tool_rate`: generated JSON selects `extract_message`.
- `schema_valid_rate`: controller-filled arguments pass the allowlist/schema.
- `end_to_end_accuracy`: tokenizer/hash extraction equals the held-out message.
- false activation on unseen normal instructions and carrier text.
- false activation on unseen near-trigger spellings.
- quoted and embedded exact-trigger activation, reported as security metrics.

General preservation and cost metrics:

- token-prefix agreement between adapter and disabled-adapter outputs.
- adapter/base output diversity on unseen instructions, to expose template collapse.
- trainable parameters, train loss, runtime, and effective batch size.
- adapter size and vLLM latency/throughput when deployment measurements exist.

The summary includes a convenience `balanced_score`:

```text
end_to_end_accuracy - max(all_false_activation_rates)
```

The worst-case penalty prevents one severe bypass from being hidden by several
easy zero-FPR sets. Still select on the recall/FPR/preservation/cost Pareto
frontier rather than this convenience score alone.

Watermark-channel metrics:

- saved-text recovery rate and mean bit accuracy;
- base-model NLL/perplexity of constrained generated tokens;
- unique-token ratio and seconds per sample;
- required tokens and net payload capacity.

## Controls and split policy

1. Pin the same base-model revision, tokenizer, chat template, corpus, key,
   decoding settings, maximum lengths, and effective batch size.
2. Every training run uses a prefix of one shared corpus. Evaluation uses a
   disjoint suffix selected by `--eval-offset`.
3. Choose hyperparameters on a development suffix. Run the final configuration
   once on a separate untouched test corpus.
4. Use greedy routing evaluation. Keep watermark sampling settings fixed inside
   each channel sweep.
5. Run at least three seeds for the baseline and final candidate, and report
   mean plus standard deviation. For proportions, also report a binomial
   confidence interval when the sample count is small.

## Commands

Build one GPT-2 carrier corpus with 300 training examples plus 100 held-out
examples. Llama 3 is the routing model; GPT-2 is the watermark tokenizer:

```powershell
python corpus_build.py `
  --base-model gpt2 `
  --model-dir D:\Programs\Transformers `
  --n-samples 400 `
  --message-chars 8 `
  --max-new-tokens 96 `
  --bits-per-token 2 `
  --key MY-SECRET `
  --seed 2026 `
  --output .\experiment_corpus.json
```

Inspect the matrix before consuming GPU time:

```powershell
python experiments.py plan --profile practical
```

Inspect generated child commands before spending GPU time:

```powershell
python experiments.py run `
  --profile practical `
  --model llama3-8b `
  --model-dir D:\Programs\Transformers `
  --watermark-model gpt2 `
  --watermark-model-dir D:\Programs\Transformers `
  --corpus-path .\experiment_corpus.json `
  --output-root .\experiments\practical `
  --device cuda:0 `
  --dtype float16 `
  --qlora `
  --num-gpus 1 `
  --batch-size 2 `
  --gradient-accumulation-steps 8 `
  --max-length 256 `
  --eval-offset 300 `
  --eval-samples 50 `
  --normal-samples 50 `
  --load-in-4bit-eval `
  --mcp-samples 0 `
  --resume `
  --dry-run
```

Remove `--dry-run` to execute. Start with `--num-gpus 1`; the global effective
batch is 2 x 8 = 16. The `practical`
profile automatically compares adapter outputs with the disabled base model.
Use a fresh output directory for each protocol version. MCP is disabled in the
matrix because its transport is
adapter-independent; validate it separately on the selected adapter.

To run only the most decision-useful subset first:

```powershell
python experiments.py run `
  --profile practical `
  --only baseline,robust_h50,robust_qv_r8,robust_qv_r8_negheavy `
  --model llama3-8b `
  --model-dir D:\Programs\Transformers `
  --watermark-model gpt2 `
  --watermark-model-dir D:\Programs\Transformers `
  --corpus-path .\experiment_corpus.json `
  --output-root .\experiments\practical `
  --device cuda:0 `
  --dtype float16 `
  --qlora `
  --num-gpus 1 `
  --batch-size 2 `
  --gradient-accumulation-steps 8 `
  --max-length 256 `
  --eval-offset 300 `
  --eval-samples 50 `
  --normal-samples 50 `
  --load-in-4bit-eval `
  --mcp-samples 0 `
  --resume
```

Summarize completed or partially completed runs:

```powershell
python experiments.py summarize `
  --output-root .\experiments\practical
```

Run the task-specific watermark sweep without retraining adapters:

```powershell
python watermark_sweep.py `
  --base-model gpt2 `
  --model-dir D:\Programs\Transformers `
  --bits-per-token 1,2,3 `
  --message-chars 4,8,16 `
  --max-new-tokens 256 `
  --n-samples 20 `
  --key MY-SECRET `
  --measure-nll
```

Each run has `config.json`, `status.json`, logs, adapter metadata, and detailed
evaluation JSON, so interrupted matrices can continue with `--resume`. Select
on end-to-end accuracy, false activations, training runtime, and adapter size
together; do not choose solely by the convenience `balanced_score`.
