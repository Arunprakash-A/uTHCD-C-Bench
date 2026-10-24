# Results

`raw/` holds the per-model evaluation outputs behind every table and figure in
the paper. Each `corruption_results.json` has the schema:

```json
{
  "model": "convnext_tiny",
  "clean":     {"top1": 97.82},                       // clean accuracy (%)
  "corrupted": {"gaussian_noise": {"1": {"top1": 2.97}, ...}, ...}  // top-1 error (%)
}
```

Layout:

| Path | Protocol |
|------|----------|
| `raw/kfold/<model>/fold_{0..4}/` | 5-fold stratified cross-validation |
| `raw/writer_independent/seed_{1,42,123}/<model>/` | writer-disjoint partitions |
| `raw/writer_dependent_70/<model>/` | fixed 70:30 split (writer-dependent) |

Regenerate the tables and heatmap from `raw/`:

```bash
python scripts/aggregate.py --raw-dir results/raw --out-dir results/tables
python scripts/plot_heatmap.py
```

This writes `tables/mce_aggregate.csv`, `tables/mce_per_corruption.{csv,json}`,
`tables/mce_writer_independent.csv`, and `tables/mce_heatmap.png`. All values use
the ratio-of-sums mCE with VGG16-BN as the reference model.
