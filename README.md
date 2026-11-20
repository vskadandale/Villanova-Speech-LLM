# Villanova-Speech-LLM

Speech LLMs for **automatic speech recognition (ASR)** and **speech translation (ST)** in English and Italian.
A frozen speech encoder ([FAMA](https://huggingface.co/FBK-MT/fama-medium-asr) or
[Whisper large-v3-turbo](https://github.com/openai/whisper)) is connected through a trained projector to a frozen LLM
([Villanova-2B](https://huggingface.co/VillanovaAI/Villanova-2B-2603) or
[EuroLLM-1.7B](https://huggingface.co/utter-project/EuroLLM-1.7B)) that is adapted with LoRA.

The code is a fork of [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM), reduced to the ASR and ST recipes behind the
results below.

## Results

### Monolingual models

One model per task and language. IT is trained and tested on Common Voice 17 Italian, EN on VoxPopuli English.

| Encoder | Projector init | LLM | ASR IT (WER ↓) | ASR EN (WER ↓) | ST IT→EN (BLEU ↑) | ST EN→IT (BLEU ↑) |
|---|---|---|---|---|---|---|
| FAMA | scratch | Villanova-2B | 7.79 | 7.44 | 56.18 | 58.30 |
| FAMA | MEUSLI | Villanova-2B | 7.56 | 6.55 | 55.66 | 57.75 |
| Whisper | MEUSLI | Villanova-2B | 6.08 | 7.09 | 56.87 | 56.22 |
| Whisper | MEUSLI | EuroLLM-1.7B | 6.31 | 6.67 | 60.86 | 62.19 |

### Bilingual models (FAMA × MEUSLI × Villanova-2B)

One model for both languages, trained and tested on `large_en_it`: Common Voice 17 + FLEURS for both languages,
plus MLS (IT) and LibriSpeech (EN), with English cut to the Italian sample counts. These numbers are not comparable
with the monolingual ones.

| Training tasks | ASR IT (WER ↓) | ASR EN (WER ↓) | ST IT→EN (BLEU ↑) | ST EN→IT (BLEU ↑) |
|---|---|---|---|---|
| ASR only | 8.45 | 11.94 | – | – |
| ST only | – | – | 49.25 | 41.26 |
| ASR + ST (multitask) | 8.70 | 11.88 | 50.95 | 41.25 |

**How to read these numbers**

- **ST references are synthetic.** The ST targets (training and test) are translations of the ASR transcripts made
  with [EuroLLM-1.7B-Instruct](https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct)
  (see `data/*_st_*_preprocessing.py`).
- **Text normalisation.** The data loader lowercases targets. WER is computed without punctuation, after Whisper's
  `BasicTextNormalizer` (`EnglishTextNormalizer` for English). BLEU is computed on normalised text.
- **Projector init.** "scratch" trains the projector from random initialisation. "MEUSLI" starts from a MEUSLI
  checkpoint (a Whisper projector + LoRA trained for that LLM). With FAMA, the projector's first layer does not fit
  FAMA's output size, so it is re-initialised and the rest is loaded.
- **Prompts.** Monolingual ASR uses a language-specific prompt ("Transcribe Italian speech to text."); bilingual and
  multitask ASR use a generic one ("Transcribe speech to text."), since the language is not given. ST prompts always
  name the direction. All prompts are in `scripts/*/conf/`.
- **Whisper truncates audio to 30 s**, which affects long utterances (about 1,200 in VoxPopuli EN). FAMA has no such
  limit.

## Repository layout

```
scripts/
  asr/
    conf/          prompts (transcribe*.yaml)
    monolingual/   one model per language: finetune_<encoder>_<projector init>_<llm>_<en|it>.sh
    bilingual/     one model for en+it, plus its per-test-set evaluation
  st/
    conf/          prompts (translate_*.yaml)
    monolingual/   one model per direction: finetune_<encoder>_<projector init>_<llm>_<en2it|it2en>.sh
    bilingual/     one model for both directions
  multitask/       one model for ASR (en, it) + ST (en→it, it→en)
  eval/            per-test-set evaluation of the bilingual ST and multitask models
src/slam_llm/
  recipe/          entry points shared by all scripts (finetune.py, inference_batch.py), config, model
  ...              SLAM-LLM library code (encoders, projectors, datasets, training loop)
data/              scripts that turn each corpus into the jsonl format below
```

Every training script has a `slurm_*.sh` wrapper next to it for Slurm clusters.

## Setup

### Environment

Python 3.12, CUDA 12.8, and the pinned packages in `requirements.txt`:

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

[`CONDA_README.md`](CONDA_README.md) has the full conda setup used for the results, and the `Dockerfile` builds the
same environment:

```bash
docker build -t villanova-speech-llm .
docker run -it --gpus all --shm-size=64g villanova-speech-llm
```

### Folder layout

The scripts expect the repository and the data side by side under `$WORK`:

```
$WORK/
  Villanova-Speech-LLM/   this repository
    models/               encoders, LLMs and MEUSLI checkpoints (below)
    exp/                  outputs, created by the scripts
  data/                   jsonl manifests (below)
```

### Models

Download these into `models/` under the names the scripts use:

| File or folder in `models/` | Source |
|---|---|
| `fama-medium-asr/` | [FBK-MT/fama-medium-asr](https://huggingface.co/FBK-MT/fama-medium-asr) |
| `large-v3-turbo.pt` | [Whisper large-v3-turbo](https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt) |
| `villanova-2b/` | [VillanovaAI/Villanova-2B-2603](https://huggingface.co/VillanovaAI/Villanova-2B-2603) |
| `eurollm-1.7b/` | [utter-project/EuroLLM-1.7B](https://huggingface.co/utter-project/EuroLLM-1.7B) |
| `meusli-eurollm-1.7b.pt` | [SpeechTek/MEUSLI_projector_v2](https://huggingface.co/SpeechTek/MEUSLI_projector_v2) (`model.pt`) |
| `meusli-villanova-2b.pt` | [SpeechTek/MEUSLI_projector_v2_Villanova_2B](https://huggingface.co/SpeechTek/MEUSLI_projector_v2_Villanova_2B) (`model.pt`) |

For example:

```bash
cd models
hf download FBK-MT/fama-medium-asr --local-dir fama-medium-asr
hf download VillanovaAI/Villanova-2B-2603 --local-dir villanova-2b
hf download utter-project/EuroLLM-1.7B --local-dir eurollm-1.7b
```

## Data

Each split is a jsonl file with one utterance per line:

```json
{"key": "Italian_it_12997_st", "source": "/path/to/12997.mp3", "target": "Massimo Conti.", "task": "st", "src_lang": "it", "tgt_lang": "en"}
```

`key`, `source` (audio path) and `target` are required. `src_lang` and `tgt_lang` select the prompt in the bilingual
and multitask setups, and `task` (`asr` or `st`) is used to split multitask results.

The scripts in `data/` build these files from the original corpora (Common Voice 17, VoxPopuli, FLEURS, MLS,
LibriSpeech, ITALIC, TED-LIUM, Europarl-ST). The ST sets are derived from the ASR sets by translating the targets
with EuroLLM-1.7B-Instruct. The training scripts read `train.jsonl`, `val.jsonl` and `test.jsonl` from these folders
in `$WORK/data/`:

| Folder | Used by |
|---|---|
| `voxpopuli_jsonl`, `cv17_Italian_it_jsonl` | monolingual ASR (EN, IT) |
| `voxpopuli_en2it_jsonl`, `cv17_Italian_it2en_jsonl` | monolingual ST (EN→IT, IT→EN) |
| `large_en_it_jsonl` | bilingual ASR |
| `large_en_it_st_jsonl` | bilingual ST |
| `large_en_it_multitask_jsonl` | multitask |
| `small_en_it_jsonl` | extra evaluation (ITALIC IT + TED-LIUM EN) |

## Training and evaluation

Each script trains one model on one GPU, decodes the test set with the last checkpoint and scores it:

```bash
export WORK=/path/to/work                      # the folder that holds Villanova-Speech-LLM/ and data/
bash scripts/asr/monolingual/finetune_fama_meusli_vnova_it.sh
# or, on a Slurm cluster (edit the account/partition lines first):
sbatch scripts/asr/monolingual/slurm_finetune_fama_meusli_vnova_it.sh
```

Outputs go to `exp/<task>_<data>_<encoder>_<projector init>_<llm>/`, e.g. `exp/ASR_cv17_it_fama_meusli_vnova/`:

```
train.log, inference.log
asr_epoch_<E>_step_<S>/            checkpoint (model.pt), decode_test_gt / decode_test_pred
FINAL_GLOBAL_WER_REPORT.txt        (FINAL_GLOBAL_BLEU_REPORT*.txt for ST)
```

The bilingual and multitask models are also evaluated per test set (each corpus of `large_en_it`, VoxPopuli EN and
`small_en_it`):

```bash
bash scripts/asr/bilingual/eval_fama_meusli_vnova_bilingual.sh                  # bilingual ASR
model=bilingual bash scripts/eval/eval_fama_meusli_vnova_per_testset.sh          # bilingual ST
model=multitask bash scripts/eval/eval_fama_meusli_vnova_per_testset.sh          # multitask
```

Results go to `exp/EVAL_<training folder>/<checkpoint>/`. Set `ckpt_path=...` to evaluate a checkpoint other than
the latest.

Configuration follows SLAM-LLM's priority: command-line overrides in the shell script > the prompt yaml in `conf/` >
the dataclass defaults in `src/slam_llm/recipe/speech_config.py`.

## Known issues and TODO

**Repetition loops at decoding time.** Decoding uses beam search (4 beams, at most 200 new tokens) with no guard
against repetition (`src/slam_llm/models/slam_model.py`, `generate`). On a handful of test utterances a model gets
stuck repeating a word or phrase until it reaches the 200-token limit (e.g. "di strati di strati di strati …").
Each such output can add close to 200 inserted words, so a few of them move the WER noticeably, especially on the
smaller VoxPopuli EN test set (1,842 utterances). Which utterances loop varies from one training run to the next.

For example, the three ASR models retrained for the results above (FAMA × scratch IT and EN, FAMA × MEUSLI IT)
gave higher WERs than earlier runs of the same models, yet without the 3–7 looping utterances of each test set,
the retrained models are as good or better:

| Model | WER, all utterances (new / earlier) | WER without looping utterances (new / earlier) |
|---|---|---|
| FAMA × MEUSLI, IT | 7.56 / 7.34 | 7.22 / 7.27 |
| FAMA × scratch, IT | 7.79 / 7.68 | 7.55 / 7.67 |
| FAMA × scratch, EN | 7.44 / 6.86 | 6.88 / 6.87 |

**TODO:** add a guard against repetition at decoding time (e.g. `no_repeat_ngram_size`, a mild
`repetition_penalty`, or a token limit tied to the audio length), check that it does not hurt correct outputs, and
re-decode every model with it. This only needs decoding with the existing checkpoints, not retraining, and has to be
applied to all models alike so that results stay comparable.

## Checkpoints

The trained models are available at: **TODO link**.

## Acknowledgements

This work was supported by Ministero delle Imprese e del Made in Italy (IPCEI Cloud DM 27 giugno 2022 – IPCEI-CL-0000007) and European Union (Next Generation EU). We acknowledge the CINECA award under the ISCRA initiative
(ISCRA-C project COBRAS), for the availability of high-performance computing resources and support.

This repository builds on [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM) (MIT license), and uses
[FAMA](https://huggingface.co/FBK-MT/fama-medium-asr), [Whisper](https://github.com/openai/whisper),
[Villanova](https://huggingface.co/VillanovaAI), [EuroLLM](https://huggingface.co/utter-project) and the
MEUSLI projector checkpoints ([EuroLLM](https://huggingface.co/SpeechTek/MEUSLI_projector_v2),
[Villanova](https://huggingface.co/SpeechTek/MEUSLI_projector_v2_Villanova_2B)).

If you use this code, please also cite SLAM-LLM:

```bibtex
@article{ma2026slam,
  title={SLAM-LLM: A Modular, Open-Source Multimodal Large Language Model Framework and Best Practice for Speech, Language, Audio and Music Processing},
  author={Ma, Ziyang and Yang, Guanrou and Chen, Wenxi and Gao, Zhifu and Du, Yexing and Li, Xiquan and Zheng, Zhisheng and Zhu, Haina and others},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  year={2026}
}
```

## License

MIT, see [LICENSE](LICENSE). Some files under `src/` come from other projects and carry their own
licence header (e.g. Llama 2 Community License, GPL-3.0); for those files, the header applies.
