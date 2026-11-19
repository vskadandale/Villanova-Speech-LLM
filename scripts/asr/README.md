# ASR recipes

Speech recognition for English and Italian. Setup, model downloads and the data format are in the
[top-level README](../../README.md).

## Models

| Script | Language | Encoder | Projector init | LLM | Prompt (`conf/`) | WER ↓ |
|---|---|---|---|---|---|---|
| `monolingual/finetune_fama_scratch_vnova_it.sh` | IT | FAMA | scratch | Villanova-2B | `transcribe_it.yaml` | 7.68 |
| `monolingual/finetune_fama_scratch_vnova_en.sh` | EN | FAMA | scratch | Villanova-2B | `transcribe_en.yaml` | 6.86 |
| `monolingual/finetune_fama_meusli_vnova_it.sh` | IT | FAMA | MEUSLI | Villanova-2B | `transcribe_it.yaml` | 7.34 |
| `monolingual/finetune_fama_meusli_vnova_en.sh` | EN | FAMA | MEUSLI | Villanova-2B | `transcribe_en.yaml` | 6.55 |
| `monolingual/finetune_whisper_meusli_vnova_it.sh` | IT | Whisper | MEUSLI | Villanova-2B | `transcribe_it.yaml` | 6.08 |
| `monolingual/finetune_whisper_meusli_vnova_en.sh` | EN | Whisper | MEUSLI | Villanova-2B | `transcribe_en.yaml` | 7.09 |
| `monolingual/finetune_whisper_meusli_euro_it.sh` | IT | Whisper | MEUSLI | EuroLLM-1.7B | `transcribe_it.yaml` | 6.31 |
| `monolingual/finetune_whisper_meusli_euro_en.sh` | EN | Whisper | MEUSLI | EuroLLM-1.7B | `transcribe_en.yaml` | 6.67 |
| `bilingual/finetune_fama_meusli_vnova_bilingual.sh` | IT + EN | FAMA | MEUSLI | Villanova-2B | `transcribe.yaml` | 8.45 (IT), 11.94 (EN) |

Data: monolingual IT uses Common Voice 17 Italian (`cv17_Italian_it_jsonl`), monolingual EN uses VoxPopuli English
(`voxpopuli_jsonl`). The bilingual model uses `large_en_it_jsonl` (Common Voice 17 + FLEURS for both languages, MLS
for IT, LibriSpeech for EN), so its WERs are not comparable with the monolingual ones.

Prompts: the monolingual models name the language ("Transcribe Italian speech to text."); the bilingual model uses
"Transcribe speech to text." for both, since the language of the audio is not given to it.

## What is trained

The encoder and the LLM are frozen. The projector and a LoRA adapter on the LLM (`q_proj`, `v_proj`) are trained.

- **FAMA × scratch**: a randomly initialised projector (`encoder_projector="linear"`: 2 stacked FAMA frames
  = 2048 inputs, a 2048 hidden layer, then the LLM size) and a new LoRA (rank 8, alpha 32).
- **FAMA × MEUSLI**: starts from the MEUSLI checkpoint for Villanova-2B (a Whisper projector and LoRA rank 32).
  Its first projector layer expects 5 stacked Whisper frames (6400 inputs) and cannot take FAMA's output, so it is
  re-initialised for 2 stacked FAMA frames (`init_skip_prefix=encoder_projector.linear1`). The rest is loaded, and
  everything is trained (`train_only_prefix=null`).
- **Whisper × MEUSLI**: starts from the MEUSLI checkpoint for the same LLM with the same Whisper encoder, so every
  tensor is loaded and training continues from there.

Shared settings: learning rate 1e-4 with 1000 warm-up steps, batch size 4, 4 epochs, fp16, one GPU. Validation runs
every 5000 steps, and a checkpoint is saved whenever the validation loss improves.

## Running

```bash
export WORK=/path/to/work     # holds Villanova-Speech-LLM/ and data/
bash scripts/asr/monolingual/finetune_fama_meusli_vnova_it.sh
sbatch scripts/asr/monolingual/slurm_finetune_fama_meusli_vnova_it.sh   # Slurm: set the account/partition lines first
```

Each script trains, then decodes `test.jsonl` with the latest checkpoint (the one with the best validation loss) and
computes WER. Outputs go to `exp/ASR_<data>_<encoder>_<projector init>_<llm>/`:

```
train.log, inference.log
asr_epoch_<E>_step_<S>/model.pt          checkpoint
asr_epoch_<E>_step_<S>/decode_test_gt    references
asr_epoch_<E>_step_<S>/decode_test_pred  hypotheses
FINAL_GLOBAL_WER_REPORT.txt
```

WER is computed without punctuation, after Whisper's `BasicTextNormalizer` (`compute_wer.py`), or its
`EnglishTextNormalizer` for English (`compute_wer_en.py`).

### Per-test-set evaluation of the bilingual model

```bash
bash scripts/asr/bilingual/eval_fama_meusli_vnova_bilingual.sh
```

It splits the existing `large_en_it` decode by corpus and scores each part, then decodes and scores VoxPopuli EN and
`small_en_it` (ITALIC IT + TED-LIUM EN). Results go to `exp/EVAL_ASR_large_en_it_fama_meusli_vnova/<checkpoint>/`;
set `ckpt_path=...` to evaluate a different checkpoint.

The multitask model (ASR + ST in one model) is in [`scripts/multitask`](../multitask).
