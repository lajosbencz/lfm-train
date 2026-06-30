# Finding: the compositional generalization wall

Empirical result from fine-tuning LFM2.5 (230M / 350M / 1.2B-Base) with QLoRA SFT
on PromQL query optimization. Established across four benchmark runs on a
held-out-vocabulary eval (train metrics/jobs disjoint from eval vocab).

## What generalizes

- **Single-transform generalization is free and scale-independent.**
  Unseen-vocabulary accuracy is 94-98% across all three model sizes. The models
  learn the transformation, not the tokens (seen-vs-unseen vocab delta ~0pp).
- **Composition is learnable per-type.** A composition *type* included in
  training reaches 91-100% on unseen vocabulary - vocab transfer is excellent.

## What does not

- **Composition does NOT generalize to unseen combinations.** A composition type
  held out of training scores 16-25%. The wall is not breached by:
  - **coverage** - training 5 composition types left the 6th at ~0-33%;
  - **contrastive instruction-sensitivity** - broke shape-routing for single
    transforms (model can be steered to either transform by instruction), but
    composition stayed ~16%;
  - **scale** - 1.2B reached only 25% vs 16% for 230M/350M; a nudge, not a phase
    change, and far below the same model's 95-100% on trained compositions.
  - **architecture** - Qwen3-0.6B-Base (different family, dense attention +
    SwiGLU, tied embeddings) scores 0% on the same held-out compositions while
    matching or beating LFM2.5 everywhere else (90% overall, 100% comp-trained).
    The wall is a property of the task, not one model family's design.

## Mechanism: multi-edit synthesis, not instruction-reading

The model emits **one structural edit per generation**. Asked to apply two
(e.g. "add a 5m downsample step AND filter to job=X"), it anchors on the first
emittable / structurally-primed edit and drops the other.

Smoking gun: a composition combining two transforms each individually ~100%
(downsample + label-add) scores **0/6 when the downsample clause is mentioned
first - on all three model sizes, including 1.2B**. Clause order matters: leading
with the fragile transform (label-insert) rescues a few cases.

The contrastive experiment rules out instruction-reading as the bottleneck: the
model *can* be steered to either single transform by instruction. The wall is
generating two simultaneous edits in one output.

## Practical implication

To get composition reliability, put the composed form directly in training - it
then transfers to unseen vocabulary at 91-100%. Do not expect composition to
emerge from component training, coverage, instruction-sensitivity tricks, or
larger models. Teach composition explicitly; expect single transforms for free.

## How this is measured

Eval is bucketed by the generation engine: `single_seen` (train vocab),
`single_unseen` (held-out vocab), `composed_trained` (composition type trained,
new vocab), `composed_heldout` (composition type never trained). The benchmark
prints these buckets directly, so re-running surfaces the wall without rederiving
it.
