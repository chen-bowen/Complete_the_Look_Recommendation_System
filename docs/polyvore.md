# Polyvore Dataset

Polyvore provides outfit compatibility data (compatible vs incompatible item pairs).

## Prepare

```bash
uv run python -m src.data_pipeline.data_preparation polyvore
```

Options in `configs/data_prep.yaml` under `polyvore`:
- `out_dir`, `max_triplets`, `download_images`, `streaming`

Set `download_images: true` to fetch images from Marqo/polyvore (~2.5GB). Otherwise add images manually from [xthan/polyvore-dataset](https://github.com/xthan/polyvore-dataset).

## Merge into Compatibility

Enable in config:

```yaml
use_polyvore_in_compatibility: true
```

Polyvore triplets are concatenated with CTL for a single compatibility loss.
