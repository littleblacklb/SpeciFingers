# Task: Theme 1（Global Average Pooling Decoder）

## Context
- Repository: SpeciFingers
- Current baseline: CNN Encoder + `DecoderRNN` (LSTM)
- Chosen theme: **Theme 1 = Global Average Pooling temporal decoder**
- Priority: simple implementation and smooth training process

## Goal
Replace the current LSTM decoder with a Global Average Pooling based decoder while keeping the existing encoder/data pipeline unchanged.

## Scope
- In scope:
  - Add a new decoder class (pooling-based classifier)
  - Add CLI option to choose decoder type
  - Integrate new decoder into training flow
  - Keep checkpoint saving and validation logic working
- Out of scope:
  - Data reprocessing
  - Multi-model benchmark suite
  - Hyperparameter search

## Required Changes
1. Add a new decoder in `functions.py`:
   - Class name suggestion: `DecoderPool`
   - Input shape: `(batch, time, embed_dim)`
   - Core op: `x.mean(dim=1)` over time dimension
   - Head: `Linear(embed_dim -> h_FC_dim) + ReLU + Dropout + Linear(h_FC_dim -> num_classes)`

2. Re-export it in `functions_optimized.py`:
   - Update import list from `functions.py` to include `DecoderPool`

3. Update `model_optimized.py`:
   - Add argument `--decoder` with choices: `lstm`, `pool`
   - Default can remain `lstm` for backward compatibility
   - If `--decoder pool`, instantiate `DecoderPool`; otherwise keep `DecoderRNN`
   - Keep optimizer parameter collection compatible with both decoders

4. Keep compatibility:
   - Existing `--encoder` options must still work
   - Existing checkpoint naming logic can remain unchanged

## Acceptance Criteria
- `python model_optimized.py --test --encoder alexnet --decoder pool` runs without code errors
- Training + validation complete at least 1 epoch in test mode
- Output shapes remain correct for cross-entropy classification (`(batch, 3)`)
- No changes required to packed data format

## Suggested Minimal Test Plan
1. Dry run:
   - `python model_optimized.py --test --encoder alexnet --decoder pool`
2. Compare with baseline startup:
   - `python model_optimized.py --test --encoder alexnet --decoder lstm`
3. Verify both can initialize, train one test epoch, and save checkpoints

## Notes for Next Conversation
- Focus on correctness and clean integration first
- Keep edits minimal and localized to:
  - `functions.py`
  - `functions_optimized.py`
  - `model_optimized.py`
- Do not change data preparation scripts unless a hard blocker appears
