# ST recipes

Speech translation between English and Italian. Setup, model downloads and the data format are in the
[top-level README](../../README.md).

## Models

| Script | Direction | Encoder | Projector init | LLM | Prompt (`conf/`) | BLEU ↑ |
|---|---|---|---|---|---|---|
| `monolingual/finetune_fama_scratch_vnova_it2en.sh` | IT→EN | FAMA | scratch | Villanova-2B | `translate_it2en.yaml` | 56.18 |
| `monolingual/finetune_fama_scratch_vnova_en2it.sh` | EN→IT | FAMA | scratch | Villanova-2B | `translate_en2it.yaml` | 58.30 |
| `monolingual/finetune_fama_meusli_vnova_it2en.sh` | IT→EN | FAMA | MEUSLI | Villanova-2B | `translate_it2en.yaml` | 55.66 |
| `monolingual/finetune_fama_meusli_vnova_en2it.sh` | EN→IT | FAMA | MEUSLI | Villanova-2B | `translate_en2it.yaml` | 57.75 |
| `monolingual/finetune_whisper_meusli_vnova_it2en.sh` | IT→EN | Whisper | MEUSLI | Villanova-2B | `translate_it2en.yaml` | 56.87 |
| `monolingual/finetune_whisper_meusli_vnova_en2it.sh` | EN→IT | Whisper | MEUSLI | Villanova-2B | `translate_en2it.yaml` | 56.22 |
| `monolingual/finetune_whisper_meusli_euro_it2en.sh` | IT→EN | Whisper | MEUSLI | EuroLLM-1.7B | `translate_it2en.yaml` | 60.86 |
| `monolingual/finetune_whisper_meusli_euro_en2it.sh` | EN→IT | Whisper | MEUSLI | EuroLLM-1.7B | `translate_en2it.yaml` | 62.19 |
| `bilingual/finetune_fama_meusli_vnova_bilingual.sh` | both | FAMA | MEUSLI | Villanova-2B | `translate_bilingual.yaml` | 49.25 (IT→EN), 41.26 (EN→IT) |

Data: IT→EN uses Common Voice 17 Italian (`cv17_Italian_it2en_jsonl`), EN→IT uses VoxPopuli English
(`voxpopuli_en2it_jsonl`). The bilingual model uses `large_en_it_st_jsonl` (Common Voice 17 + FLEURS for both
languages, MLS for IT, LibriSpeech for EN), so its scores are not comparable with the monolingual ones.

**The translations are synthetic.** These sets are the ASR sets with each transcript replaced by its
EuroLLM-1.7B-Instruct translation (`data/*_st_*_preprocessing.py`), for training and test alike.

Prompts name the direction ("Translate English speech to Italian text. "). The bilingual model picks the prompt of
each record from its `src_lang` and `tgt_lang`.

## What is trained

The encoder and the LLM are frozen. The projector and a LoRA adapter on the LLM (`q_proj`, `v_proj`) are trained.

- **FAMA × scratch**: a randomly initialised projector (`encoder_projector="linear"`: 2 stacked FAMA frames
  = 2048 inputs, a 2048 hidden layer, then the LLM size) and a new LoRA (rank 8, alpha 32), as in ASR.
- **FAMA × MEUSLI**: starts from the MEUSLI checkpoint for Villanova-2B (a Whisper projector and LoRA rank 32).
  Its first projector layer expects 5 stacked Whisper frames (6400 inputs) and cannot take FAMA's output, so it is
  re-initialised for 2 stacked FAMA frames (`init_skip_prefix=encoder_projector.linear1`). The rest is loaded, and
  everything is trained (`train_only_prefix=null`).
- **Whisper × MEUSLI**: starts from the MEUSLI checkpoint for the same LLM with the same Whisper encoder, so every
  tensor is loaded and training continues from there.

Shared settings: learning rate 1e-4 with 1000 warm-up steps, 4 epochs, fp16, one GPU, validation every 5000 steps,
and a checkpoint is saved whenever the validation loss improves. Batch size is 4, except for EN→IT with FAMA and
with Whisper × Villanova and for the bilingual model, which use batch size 2 with 2 gradient-accumulation steps
(same effective batch) to fit in memory.

## Running

```bash
export WORK=/path/to/work     # holds Villanova-Speech-LLM/ and data/
bash scripts/st/monolingual/finetune_fama_meusli_vnova_en2it.sh
sbatch scripts/st/monolingual/slurm_finetune_fama_meusli_vnova_en2it.sh   # Slurm: set the account/partition lines first
```

Each script trains, then decodes `test.jsonl` with the latest checkpoint (the one with the best validation loss) and
computes BLEU. Outputs go to `exp/ST_<data>_<encoder>_<projector init>_<llm>/`:

```
train.log, inference.log
st_epoch_<E>_step_<S>/model.pt          checkpoint
st_epoch_<E>_step_<S>/decode_test_gt    references
st_epoch_<E>_step_<S>/decode_test_pred  hypotheses
FINAL_GLOBAL_BLEU_REPORT.txt            (one per direction for the bilingual model)
```

BLEU is computed after Whisper's text normalisation: `compute_bleu.py` for Italian targets, `compute_bleu_en.py`
for English targets.

### Per-test-set evaluation of the bilingual model

```bash
model=bilingual bash scripts/eval/eval_fama_meusli_vnova_per_testset.sh
```

It splits the existing `large_en_it` decode by corpus and direction and scores each part, then decodes and scores
VoxPopuli EN→IT. Results go to `exp/EVAL_ST_large_en2it_it2en_fama_meusli_vnova/<checkpoint>/`; set
`ckpt_path=...` to evaluate a different checkpoint.

The multitask model (ASR + ST in one model) is in [`scripts/multitask`](../multitask) and is evaluated with
`model=multitask` in the same script.
