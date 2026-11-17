TRAINING DATA

*********ITALIAN*********
[1] CV-17 (.MP3, 32/48kHz)
Train: 339.542/2 = 169.771 samples (249:55:10)
Dev: 30.298/2 = 15.149 samples (25:18:13)
Test: 30.310/2 = 15.155 samples (26:27:01)

[2] MLS (.FLAC, 16kHz)
Train: 59.623 samples (247:22:40)
Dev: 1.248 samples (05:10:32)
Test: 1.262 samples (05:16:22)

[3] FLEURS (.WAV, 16kHz)
Source: https://huggingface.co/datasets/google/fleurs/viewer/it_it
Train: 3.030 samples (08:59:53)
Dev: 391 samples (01:32:48)
Test: 865 samples (03:31:16)

*********ENGLISH*********
[1] CV-17 (.MP3, 32/48kHz)
Train: 2.202.340/2 = 1.101.170 samples (1738:29:49)
Dev: 32.786/2 = 16.393 samples (27:16:01)
Test: 32.786/2 = 16.393 samples (26:57:47)

[2] LibriSpeech-960h (.FLAC, 16kHz)
Train: 281.241 samples (496:51:28+363:36:20+100:35:27 = 961:03:15)
Dev: 5.567 samples (10:30:32, dev_other + dev_clean)
Test: 2.620 samples (05:24:12, test_clean)

[3] FLEURS (.WAV, 16kHz)
Source: https://huggingface.co/datasets/google/fleurs/viewer/en_us
Train: 2.602 samples (07:29:26)
Dev: 394 samples (01:02:45)
Test: 647 samples (01:46:27)

-------------------------------------------------------------------------------------------------------
COMBINED BILINGUAL TRAINING DATA
1. CV17 (en, it): mix top {169.771 (train), 15.149(dev), 15.155(test)}

2. FLEURS (en, it): mix top {2.602 (train), 391 (dev),  647(test)}

3. MLS (it): all
LibriSpeech (en): top {59.623 (train), 1.248 (dev), 1262 (test)}

cat <(head -n 169771 /stek/vskadandale/data/cv17_Italian_it_jsonl/train.jsonl) <(head -n 169771 /stek/vskadandale/data/cv17_English_en_jsonl/train.jsonl) <(head -n 2602 /stek/vskadandale/data/fleurs_it_it_jsonl/train.jsonl) <(head -n 2602 /stek/vskadandale/data/fleurs_en_us_jsonl/train.jsonl) <(head -n 59623 /stek/vskadandale/data/mls_italian_jsonl/train.jsonl) <(head -n 59623 /stek/vskadandale/data/librispeech_jsonl/train.jsonl) | shuf > /stek/vskadandale/data/meusli_fama_villanova_bilingual_jsonl/train.jsonl
cat <(head -n 15149 /stek/vskadandale/data/cv17_Italian_it_jsonl/validation.jsonl) <(head -n 15149 /stek/vskadandale/data/cv17_English_en_jsonl/validation.jsonl) <(head -n 391 /stek/vskadandale/data/fleurs_it_it_jsonl/dev.jsonl) <(head -n 391 /stek/vskadandale/data/fleurs_en_us_jsonl/dev.jsonl) <(head -n 1248 /stek/vskadandale/data/mls_italian_jsonl/dev.jsonl) <(head -n 1248 /stek/vskadandale/data/librispeech_jsonl/dev.jsonl) | shuf > /stek/vskadandale/data/meusli_fama_villanova_bilingual_jsonl/val.jsonl
cat <(head -n 15155 /stek/vskadandale/data/cv17_Italian_it_jsonl/test.jsonl) <(head -n 15155 /stek/vskadandale/data/cv17_English_en_jsonl/test.jsonl) <(head -n 647 /stek/vskadandale/data/fleurs_it_it_jsonl/test.jsonl) <(head -n 647 /stek/vskadandale/data/fleurs_en_us_jsonl/test.jsonl) <(head -n 1262 /stek/vskadandale/data/mls_italian_jsonl/test.jsonl) <(head -n 1262 /stek/vskadandale/data/librispeech_jsonl/test-clean.jsonl) | shuf > /stek/vskadandale/data/meusli_fama_villanova_bilingual_jsonl/test.jsonl

-------------------------------------------------------------------------------------------------------
EVALUATION DATA 
ITALIAN
- Europarl-ST
- ITALIC
- IEMOCAP/EMOVO/CREMA-D

ENGLISH
- Europarl-ST
- VoxCeleb
- AMI Meeting Corpus
- TED-LIUM v3 (test)