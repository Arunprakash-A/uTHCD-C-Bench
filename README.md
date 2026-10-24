# uTHCD-C

A corruption-robustness benchmark for **Tamil handwritten character recognition**,
built on the uTHCD dataset. It adds **ten corruption types at five severity
levels** (nine general image corruptions plus a handwriting-specific
stroke-thinning operator) and scores models with the **mean Corruption Error
(mCE)**, normalized against a VGG16-BN reference using the ImageNet-C
ratio-of-sums convention.

This repository contains the corruption suite, the evaluation/aggregation code,
and the raw per-model results behind every table and figure in the paper.

## Install

```bash
pip install -r requirements.txt
```

## Reproduce the paper's results

The raw per-model outputs are included under `results/raw/`, so the tables and
the heatmap regenerate without a GPU or the image dataset:

```bash
python scripts/aggregate.py        # mCE tables -> results/tables/*.csv
python scripts/plot_heatmap.py     # heatmap    -> results/tables/mce_heatmap.png
```

Aggregate mCE (5-fold, ratio-of-sums; lower is better):

| Model | mCE | | Model | mCE |
|---|---|---|---|---|
| ConvNeXt | 0.75 | | SqueezeNet | 1.10 |
| Swin | 0.83 | | RegNet | 1.25 |
| EfficientNet | 0.91 | | ShuffleNet | 1.55 |
| GoogLeNet | 0.98 | | MnasNet | 1.85 |
| **VGG16-BN** | **1.00** | | | |

## Benchmark your own model

1. Prepare the uTHCD test split as an `ImageFolder` (see [`data/README.md`](data/README.md)).
2. Evaluate the model over the full corruption suite:

   ```bash
   python scripts/evaluate_model.py \
       --data-dir  /path/to/uTHCD/test \
       --arch      convnext_tiny \
       --checkpoint /path/to/model.pt \
       --output    results/my_model/corruption_results.json \
       --baseline  results/raw/kfold/vgg16_bn/fold_0/corruption_results.json
   ```

   The command prints the clean accuracy, the per-corruption errors, and the
   mCE against the reference. For a custom architecture, replace `--arch` with
   `--model-script my_model.py`, where the file exposes `build() -> nn.Module`.

3. Compare models by placing their result files alongside the reference and
   re-running `scripts/aggregate.py`.

## Layout

```
uthcd_c/        corruption suite, mCE metric, model builders, data loader
scripts/        evaluate_model.py · aggregate.py · plot_heatmap.py
results/raw/    per-model results for all protocols (inputs to every table/figure)
data/           dataset instructions and expected layout
```

## Citation

Preethi S, Meeradevi T, Arun Prakash A. *uTHCD-C: A Corruption-Robustness
Benchmark for Tamil Handwritten Character Recognition.*

- Preethi S, Meeradevi T — Department of Electronics and Communication,
  Kongu Engineering College, Erode, Tamil Nadu, India
- Arun Prakash A — AI4Bharat, Indian Institute of Technology Madras, Chennai, India

## License

Code and derived results are released under the [MIT License](LICENSE). The
uTHCD image dataset is distributed separately under its own terms.
