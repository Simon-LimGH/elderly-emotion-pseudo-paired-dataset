# Pseudo-Paired Elderly Emotion Dataset (Face + Text)

A synthetic multimodal dataset that attaches a real text sentence to each labelled
face image, matched **only** on the emotion label. It was built for the study
*“Elderly Emotion Detection Using a Multimodal Approach”* to demonstrate and
evaluate an age-aware late-fusion pipeline, because no public dataset pairs an
elderly **face + age + emotion + text** from the same individual.

> ⚠️ **This is a pseudo-paired (synthetic) set.** The face and the text do **not**
> come from the same person; they share only an emotion label. Any text-only or
> fused metric measured on it is therefore **optimistic** and should be read as a
> proof of concept, not a clean benchmark.

## What is in this repo

```
PseudoPaired_Dataset/
├── build_pseudo_paired_dataset.py   # reproducible builder (face manifest + GoEmotions -> paired CSVs)
├── requirements.txt
├── faces_manifest/                  # face references + labels (NO images, NO text)
│   ├── train.csv   val.csv   test.csv
└── pseudo_paired/                   # the dataset: manifest + a paired `text` column
    ├── train.csv   val.csv   test.csv
```

**Face images are intentionally NOT included** (see *Images & licensing* below).
The CSVs reference the faces by relative path so the dataset can be committed
without redistributing anyone’s photograph.

### CSV columns (`pseudo_paired/*.csv`)

| column | description |
|---|---|
| `source` | origin dataset of the face: `ffhq`, `imdb`, or `utkface` |
| `image_relpath` | path to the (pre-processed) face crop, relative to the dataset root |
| `filename` | the image file name only |
| `emotion_label` | integer 0–7 |
| `emotion_name` | Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise |
| `text` | a GoEmotions sentence expressing the same emotion (empty for **Contempt**) |

`faces_manifest/*.csv` is identical but without the `text` column.

### Splits

| split | rows | empty text (Contempt) |
|---|---|---|
| train | 6,662 | 54 |
| val   | 1,021 | 19 |
| test  | 2,153 | 28 |

Emotion label map: `0 Anger · 1 Contempt · 2 Disgust · 3 Fear · 4 Happiness · 5 Neutral · 6 Sadness · 7 Surprise`.

## How it was built

1. Faces are the pre-processed, emotion-labelled crops (aged 50+) from **UTKFace**,
   **FFHQ**, and **IMDB-Clean**, using the 8-class scheme above.
2. **GoEmotions** comments are filtered to single-label examples, the 27 emotions
   are collapsed to the 7 Ekman emotions, and each is mapped to one of the 8 facial
   slots. There is no *Contempt* text slot, so Contempt faces keep an empty text.
3. For every face, a random same-emotion sentence is drawn from the matching split’s
   text pool (with replacement only when a pool is smaller than the number of faces).
   A fixed seed (`42`) and the ordered pools make the pairing deterministic.

## Reproduce

```bash
pip install -r requirements.txt
python build_pseudo_paired_dataset.py
```

This downloads GoEmotions from the Hugging Face Hub and regenerates
`pseudo_paired/*.csv` from `faces_manifest/*.csv`. Given the same GoEmotions
release, the output matches the committed CSVs (which are the canonical version).

## Images & licensing

The paired **text** is from **GoEmotions** (Google Research), released under the
**Apache License 2.0**, and is included here with attribution.

The **face images are not redistributed** in this repo — they are real people’s
photographs and each source dataset carries its own non-commercial / research-only
terms:

- **UTKFace** — https://susanqq.github.io/UTKFace/ (research use)
- **FFHQ** — https://github.com/NVlabs/ffhq-dataset (CC BY-NC-SA; per-image Flickr licenses)
- **IMDB-Clean / IMDB-WIKI** — Y. Lin, J. Shen, Y. Wang, and M. Pantic, "FP-Age: Leveraging face parsing attention for facial age estimation in the wild," 

To obtain the faces, download the original datasets from the links above and apply
the pre-processing used in this project; the `image_relpath` / `filename` columns
identify each crop. Please respect each dataset’s license and do not commit the raw
images to a public repository.

## Citation

If you use this dataset, please cite the source corpora:

- D. Demszky et al., “GoEmotions: A Dataset of Fine-Grained Emotions,” ACL, 2020.
- Z. Zhang, Y. Song, H. Qi, “UTKFace: Large Scale Face Dataset.”
- T. Karras et al., “A Style-Based Generator Architecture for GANs (FFHQ),” CVPR, 2019.
- Y. Lin et al., “FP-Age: Facial Age Estimation in the Wild (IMDB-Clean),” arXiv:2106.11145, 2021.
