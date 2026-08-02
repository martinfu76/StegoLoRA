# StegoLoRA

StegoLoRA is a research prototype for trigger-routed stego/watermark extraction:

1. A hash-based channel embeds a framed payload into generated token IDs.
2. A LoRA/QLoRA adapter learns when to emit an `extract_message` tool call.
3. A controller validates the call and supplies runtime-only arguments.
4. A tokenizer-only MCP server extracts the payload.

The adapter does not learn the hidden message, key, carrier text, or extraction
algorithm. Its job is routing and tool-call formatting.

## Architecture

```text
Sender: prompt + message + key
          -> stego/watermark carrier model
          -> carrier text with an embedded payload

Receiver: <|EXTRACT|> + watermarked text
          -> Llama + LoRA router (local or remote vLLM)
          -> validated extract_message call
          -> MCP server + carrier tokenizer
          -> hidden message
```

The router and carrier may be different models. The reference configuration
uses Llama 3 8B for routing and GPT-2 for stego/watermark generation and
extraction. The generic tool target is `extract_message`; the currently bundled
backend is the framed hash-based watermark channel.

CLI options accept both the generalized `--carrier-model` /
`--carrier-model-dir` names and the historical `--watermark-model` /
`--watermark-model-dir` names. They are exact aliases, so existing scripts and
saved commands remain compatible.

## Repository Layout

| File | Responsibility |
| --- | --- |
| `pipeline.py` | End-to-end training, send, receive, vLLM, and verification CLI |
| `train.py`, `data.py` | LoRA/QLoRA dataset construction and training |
| `corpus_build.py` | Build verified stego/watermark training carriers |
| `hash_watermark.py`, `embed.py` | Framed payload embedding and extraction |
| `agent.py`, `mcp_server.py` | Generic extraction controller and tokenizer-only MCP receiver |
| `vllm_server.py`, `vllm_agent.py` | Ubuntu vLLM server and remote receiver client |
| `deploy_vllm_linux.sh` | Native Conda/vLLM launcher for Ubuntu L40 |
| `evaluate.py`, `experiments.py` | Trigger, false-activation, and ablation evaluation |
| `watermark_sweep.py` | Capacity, recovery, and carrier-quality experiments |
| `generate_examples.py` | Paired plain/watermarked paper examples |
| `project_paths.py` | Shared defaults for generated output directories |

`extractor.py` is used only for synthetic fallback samples. Build a real corpus
with `corpus_build.py` before reporting end-to-end watermark results.

## Installation

Run scripts from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. QLoRA requires a CUDA build
of PyTorch and working `bitsandbytes`. Gated Hugging Face models require
`HUGGINGFACE_HUB_TOKEN` or a complete local snapshot.

Useful environment variables:

```text
STEGOLORA_MODEL_DIR       local model root
STEGOLORA_KEY             sender/receiver watermark key
HUGGINGFACE_HUB_TOKEN     gated model access token
VLLM_BASE_URL             remote OpenAI-compatible endpoint
VLLM_API_KEY              remote API key
STEGOLORA_OUTPUT_DIR      generated-output root (default: ./outputs)
```

## Generated Outputs

When a path is not supplied explicitly, generated files are kept outside the
source tree under `outputs/`:

```text
outputs/
  adapters/       LoRA/QLoRA adapters and training metadata
  corpora/        generated watermark corpora
  experiments/    per-profile training/evaluation runs and summaries
  runtime/        send/receive text and pipeline state
```

`outputs/` is ignored by Git. Set `STEGOLORA_OUTPUT_DIR` to move the whole tree,
for example to a large local data disk. Experiment CSV/JSON files and model
artifacts remain local; the README contains only the reportable result summary.

## QLoRA Training

Build a verified GPT-2 carrier corpus and train a Llama 3 QLoRA router:

```powershell
python pipeline.py prepare --model llama3-8b --model-dir D:\models --watermark-model gpt2 --watermark-model-dir D:\models --key MY-SECRET --corpus-path .\outputs\corpora\stego_corpus.json --corpus-size 600 --message-chars 8 --max-new-tokens 96 --bits-per-token 2 --n-positive 300 --n-negative 300 --hard-negative-fraction 0.5 --epochs 3 --batch-size 2 --gradient-accumulation-steps 8 --lr 2e-4 --lora-r 8 --lora-alpha 16 --target-modules q_proj,v_proj --max-length 256 --prompt-format auto --qlora --device cuda:0 --adapter-path .\outputs\adapters\stegolora
```

Reuse an existing corpus:

```powershell
python pipeline.py prepare --skip-corpus --model llama3-8b --model-dir D:\models --watermark-model gpt2 --watermark-model-dir D:\models --corpus-path .\outputs\corpora\stego_corpus.json --n-positive 300 --n-negative 300 --hard-negative-fraction 0.5 --epochs 3 --batch-size 2 --gradient-accumulation-steps 8 --lora-r 8 --lora-alpha 16 --target-modules q_proj,v_proj --qlora --device cuda:0 --adapter-path .\outputs\adapters\stegolora
```

QLoRA uses a 4-bit NF4 base during training. The output is a standard PEFT
adapter.

## Local End-to-End Run

Generate a GPT-2 carrier:

```powershell
python pipeline.py send --model llama3-8b --model-dir D:\models --watermark-model gpt2 --watermark-model-dir D:\models --message HELLO --key MY-SECRET --bits-per-token 2 --max-new-tokens 96
```

Load Llama and the adapter locally, then invoke MCP:

```powershell
python pipeline.py receive --model llama3-8b --model-dir D:\models --watermark-model gpt2 --watermark-model-dir D:\models --key MY-SECRET --bits-per-token 2 --prompt-format auto --verbose
```

Verify:

```powershell
python pipeline.py verify
```

Use `receive --direct` to bypass MCP transport while debugging. Use
`receive --no-mcp --verbose` to inspect only the raw model output.

## Ubuntu vLLM Deployment

The supported server target is Ubuntu with an NVIDIA GPU. Copy the
Llama model and PEFT adapter to the server. vLLM runs directly in a dedicated
Conda environment; Docker is not required.

Verify GPU access:

```bash
nvidia-smi
```

Create the server environment. Let the vLLM wheel install its matching PyTorch
and CUDA Python dependencies; do not preinstall a different PyTorch build into
this environment:

```bash
conda create -n stegolora-vllm python=3.12 -y
conda activate stegolora-vllm
python -m pip install --upgrade pip
python -m pip install -r requirements-vllm.txt
python -c "import torch, vllm; print(vllm.__version__, torch.__version__, torch.cuda.get_device_name(0))"
```

Launch vLLM from the activated environment:

```bash
chmod +x deploy_vllm_linux.sh
conda activate stegolora-vllm
MODEL_PATH=/srv/models/llama3-8b ADAPTER_PATH=/srv/adapters/robust_qv_r8 VLLM_API_KEY='replace-with-a-long-random-secret' ./deploy_vllm_linux.sh
```

The server exposes `stegolora-base` without the adapter and `stegolora` with
the trigger-routing adapter. Check it with:

```bash
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
```

Restrict TCP 8000 to the receiver or private network. For Internet-facing
access, put vLLM behind a VPN or HTTPS reverse proxy. A host firewall example
that only permits one receiver is:

```bash
sudo ufw allow from RECEIVER_IP to any port 8000 proto tcp
```

Call from the receiver while keeping GPT-2 and MCP local:

```powershell
$env:VLLM_API_KEY="replace-with-a-long-random-secret"; python pipeline.py receive-vllm --base-url http://L40_SERVER_IP:8000/v1 --api-key $env:VLLM_API_KEY --served-model stegolora --model llama3-8b --model-dir D:\models --watermark-model gpt2 --watermark-model-dir D:\models --key MY-SECRET --bits-per-token 2 --prompt-format auto --verbose
```

The remote server only generates the tool call. The controller inserts the
carrier text and key at runtime, then invokes the local MCP server.

## Experiments

### Experimental Setup

The completed `practical` profile run on NVIDIA `V100` 32GB GPU used Llama 3 8B as the QLoRA router and GPT-2
as the watermark carrier/tokenizer. Unless varied by the experiment, training
used 300 positive and 300 negative examples, 3 epochs, learning rate `2e-4`,
effective batch size 16, and seed 42. Evaluation used 50 triggered carriers and
50 samples for each negative category:

- normal prompts;
- watermarked carriers without the trigger;
- unseen near-trigger spellings;
- quoted exact triggers;
- exact triggers embedded later in a normal request.

All seven runs completed without Traceback, OOM, or NaN. MCP samples were set to
zero in this matrix, so the reported end-to-end metric uses direct extraction
with the same tokenizer/hash implementation as the MCP tool.

### Experimental Results

| Experiment | End-to-end | Worst false activation | Normal prefix preservation | Dominant normal template | Adapter | Train time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 100% | 88% | 0% | 50% | 160 MB | 11.1 min |
| `robust_h50` | 100% | 20% | 0% | 78% | 160 MB | 11.0 min |
| `robust_qv` | 100% | 28% | 0% | 96% | 26 MB | 8.2 min |
| **`robust_qv_r8`** | **100%** | **0%** | 0% | 98% | **13 MB** | 8.2 min |
| `robust_qv_r8_small` | 100% | 30% | 0% | 8% | 13 MB | 4.2 min |
| `robust_qv_r8_epoch1` | 98% | 98% | 13.1% | 4% | 13 MB | 2.6 min |
| `robust_qv_r8_negheavy` | 100% | 8% | 0% | 30% | 13 MB | 10.6 min |

`robust_qv_r8` modifies only `q_proj` and `v_proj`, with rank 8 and about 3.41
million trainable parameters. It achieved:

- 50/50 valid tool routes and correct direct extractions;
- 0/50 false activations on normal prompts;
- 0/50 on carriers without the trigger;
- 0/50 on unseen near triggers;
- 0/50 on quoted triggers;
- 0/50 on embedded triggers.

This is the recommended adapter for a dedicated extraction router. It is not a
good general-purpose assistant: on 50 normal evaluations it produced only two
unique outputs, and one fixed generic completion accounted for 98% of them.
The fixed negative completions taught the model to replace ordinary answering
with a template. A shared assistant/router should instead train negative
examples against base-model responses, for example through behavior
distillation.

These results are preliminary: every configuration used one seed, each safety
category had only 50 samples, the normal set contained 30 unique prompts and 20
cycled repeats, and the matrix did not execute MCP transport. Zero failures out
of 50 is evidence for this test set, not proof of a zero population error rate;
the rule-of-three upper bound is about 6% for an unseen failure probability.

### Rank, Layers, and Dropout Ablation

The later compact ablation fixed 300 positive/300 negative examples, 50% hard
negatives, 3 epochs, LR `2e-4`, seed 42, and `alpha/r=2`. All seven adapters
achieved 100% tool routing, schema validity, and direct end-to-end extraction.
Differences appeared in unseen near-trigger rejection and normal-output
preservation:

| Run | Trainable params | Adapter | Near-trigger FP | Dominant normal template | Train time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dropout_0` | 3.41M | 13.0 MB | **2% (1/50)** | 96% | 7.8 min |
| **`rank_r4`** | **1.70M** | **6.5 MB** | **8% (4/50)** | 54% | 7.9 min |
| `layers_all_linear` | 20.97M | 80.1 MB | 16% (8/50) | 54% | 11.2 min |
| `layers_attention` | 6.82M | 26.0 MB | 20% (10/50) | 72% | 9.1 min |
| `dropout_10` | 3.41M | 13.0 MB | 22% (11/50) | 54% | 8.3 min |
| `rank_r16` | 6.82M | 26.0 MB | 22% (11/50) | **50%** | 8.4 min |
| `lora_ref` (r=8) | 3.41M | 13.0 MB | 26% (13/50) | 98% | 8.4 min |

`rank_r4` is the best compact operating point in this matrix: it retains 100%
trigger extraction with half the reference parameters and substantially fewer
near-trigger activations. Expanding to all linear layers reduced training loss
but used 12.3x as many parameters as r=4 and generalized worse on near triggers.
Training loss is therefore not a reliable adapter-selection metric here.

No tested adapter preserved general answering behavior: normal-prefix agreement
was 0% for every run, adapter output diversity was 4-10% versus 60% for the
base, and 50-98% of normal prompts collapsed to one completion. This follows
from the six fixed generic negative completions in `data.py`. These experiments
validate a dedicated extraction router, not a shared assistant. Base-response
distillation or a KL-preservation objective is required before claiming normal
capability preservation.

The nominally similar r=8 result in the earlier practical matrix had 0/50 near-
trigger activations, while this later corpus/evaluation produced 13/50. That
difference is evidence of corpus/evaluation sensitivity. Final claims should
use at least three seeds, a fixed test corpus, 200+ near-trigger cases, and an
MCP-enabled run. MCP accuracy is blank here because `--mcp-samples 0` was used.

### Reproduce the Matrix

Run the practical adapter comparison:

```powershell
python experiments.py run --profile practical --model llama3-8b --model-dir D:\models --watermark-model gpt2 --watermark-model-dir D:\models --corpus-path .\outputs\corpora\experiment_corpus.json --device cuda:0 --dtype float16 --qlora --num-gpus 1 --batch-size 2 --gradient-accumulation-steps 8 --max-length 256 --eval-offset 300 --eval-samples 50 --normal-samples 50 --load-in-4bit-eval --mcp-samples 0 --resume
```

Add `--dry-run` to generate commands without executing. See
[EXPERIMENTS.md](EXPERIMENTS.md) for metrics, corpus preparation, and channel
sweeps.

For the compact rank/layers/dropout comparison:

```powershell
python experiments.py run --profile lora_ablation --model llama3-8b --model-dir D:\models --watermark-model gpt2 --watermark-model-dir D:\models --corpus-path .\outputs\corpora\experiment_corpus.json --device cuda:0 --dtype float16 --qlora --num-gpus 1 --batch-size 2 --gradient-accumulation-steps 8 --max-length 256 --eval-offset 300 --eval-samples 50 --normal-samples 50 --load-in-4bit-eval --mcp-samples 0 --resume
```

## Scope

This is research code, not a security boundary. Trigger secrecy is not
authentication. Keep keys outside prompts and logs, validate tool arguments,
restrict the vLLM port, and evaluate false activation on held-out prompts.
