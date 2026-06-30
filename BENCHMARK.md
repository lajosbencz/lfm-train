# Benchmark results

Six small base language models fine-tuned with identical QLoRA SFT on the same
PromQL query-optimization dataset, then evaluated on a held-out-vocabulary eval
set. All runs on a single RTX 3060 Ti (8GB).

Reproduce: `make benchmark` (writes `outputs/benchmark.json` + the chart below).

![benchmark chart](docs/assets/benchmark_chart.png)

## Accuracy (exact match, %)

Eval buckets, in increasing difficulty:
- **seen-vocab** - single transform, vocabulary seen in training (upper bound).
- **unseen-vocab** - single transform, held-out vocabulary (vocab transfer).
- **comp-trained** - a two-transform composition whose TYPE was trained, on
  held-out vocab (composition vocab transfer).
- **comp-heldout** - a composition TYPE never seen in training (true
  compositional generalization).

| Model | Base license | Overall | seen-vocab | unseen-vocab | comp-trained | comp-heldout |
|---|---|---:|---:|---:|---:|---:|
| LFM2.5-1.2B-Base    | LFM Open v1.0 | 92 | 100 | 98 |  95 | **25** |
| Qwen3-0.6B-Base     | Apache-2.0    | 90 |  97 | 98 | 100 | **0**  |
| h2o-danube3-500m    | Apache-2.0    | 83 |  97 | 92 |  75 | **0**  |
| SmolLM2-360M        | Apache-2.0    | 82 | 100 | 86 |  83 | **0**  |
| LFM2.5-350M-Base    | LFM Open v1.0 | 90 | 100 | 94 | 100 | **16** |
| LFM2.5-230M-Base    | LFM Open v1.0 | 90 |  97 | 96 |  95 | **16** |

## Cost (single RTX 3060 Ti, 8GB)

Training time and GPU peaks are for the full per-model run (epochs differ per
config: 230M = 12, others = 5-6). Eval throughput is tokens generated per second
over the 150-example eval set at `max_new_tokens=128`, greedy.

| Model | Size | Train time | Train it/s | Eval tok/s | Train GPU peak | Eval GPU peak | LoRA |
|---|---|---:|---:|---:|---:|---:|---|
| LFM2.5-1.2B-Base    | 1.2B | 109s | 2.25 |  75 | 1.4GB | 1.8GB | r=64 |
| Qwen3-0.6B-Base     | 0.6B | 157s | 1.87 |  78 | 3.5GB | 2.8GB | r=48 + head/embed |
| h2o-danube3-500m    | 0.5B | 113s | 2.60 | 145 | 849MB | 1.0GB | r=48 |
| SmolLM2-360M        | 360M | 120s | 2.45 |  84 | 722MB | 712MB | r=48 |
| LFM2.5-350M-Base    | 350M |  56s | 5.23 |  80 | 651MB | 690MB | r=48 |
| LFM2.5-230M-Base    | 230M |  76s | 7.70 |  86 | 525MB | 527MB | r=48 |

Notes:
- **Qwen3-0.6B** needs `modules_to_save: [lm_head, embed_tokens]` (its ChatML
  `<|im_end|>` is reserved-but-unexercised in pretraining and the embeddings are
  tied), which is why its training GPU peak is ~3.5GB vs <1.5GB for the others.
  The published artifact is a plain 0.6B model, so serving cost is small.
- **h2o-danube3-500m** has the best eval throughput by a wide margin (145 tok/s)
  but the weakest comp-trained transfer (75%).

## Takeaways

1. **Single-transform generalization is basically free** and independent of model
   size and architecture: unseen-vocab accuracy is 86-98% everywhere, with a near-
   zero seen-vs-unseen gap. The models learn the transformation, not the tokens.

2. **The compositional-generalization wall is universal.** Every model scores
   0-25% on composition types it never saw, while scoring 75-100% on the same
   compositions once the TYPE is in training. This holds across two model families
   and a 5x size range, so it is a property of the task, not of any one model.
   Larger / same-family helps only marginally (LFM2.5: 16% -> 25% from 350M to
   1.2B); cross-family Apache models sit at 0%. Full analysis:
   [docs/compositional-generalization.md](docs/compositional-generalization.md).

3. **For a license-clean publishable model, Qwen3-0.6B is the strongest** (90%
   overall, 100% comp-trained, Apache-2.0). LFM2.5-350M matches it on overall
   accuracy at a third of the train time and a fifth of the GPU, and is the
   project's flagship (LFM Open License).
