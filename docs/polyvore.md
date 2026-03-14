# Polyvore Dataset

Polyvore provides outfit compatibility data (compatible vs incompatible item pairs).

## Prepare

```bash
# Metadata only (triplets.csv); add images manually
uv run python -m src.data_pipeline.data_preparation polyvore --out-dir src/dataset/data/polyvore --max-triplets 10000

# With images (~2.5GB, from Marqo/polyvore on Hugging Face)
uv run python -m src.data_pipeline.data_preparation polyvore --download-images --out-dir src/dataset/data/polyvore
```

Without `--download-images`, only `triplets.csv` is produced. Images can be added manually from [xthan/polyvore-dataset](https://github.com/xthan/polyvore-dataset), or use `--download-images` to fetch from Hugging Face.

## Merge into Compatibility

Enable in config:

```yaml
use_polyvore_in_compatibility: true
```

Polyvore triplets are concatenated with CTL for a single compatibility loss.
