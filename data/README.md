# Dataset

uTHCD-C is built on **uTHCD** (unconstrained Tamil Handwritten Character
Database): 156 character classes, ~91,000 isolated handwritten samples at a
native resolution of 64×64. The image dataset is **not redistributed here**; it
is released separately by its original authors:

> N. Shaffi and F. Hajamohideen, "uTHCD: A New Benchmarking for Tamil
> Handwritten OCR," *IEEE Access*, vol. 9, pp. 101469–101493, 2021.
> doi:10.1109/ACCESS.2021.3096823

Obtain uTHCD from the original release and arrange the split you want to
evaluate as an `ImageFolder` (one sub-directory per class):

```
<data-dir>/
├── 000/   img_0001.png ...
├── 001/   ...
└── 155/
```

Point the evaluation script at this directory with `--data-dir <data-dir>`.
Corruptions are applied to each image at its native resolution before the
network's resize/normalization, so no manual pre-resizing is required.
