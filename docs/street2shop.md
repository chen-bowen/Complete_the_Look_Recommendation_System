# Street2Shop Dataset

Street2Shop provides street photo to shop product matching for real-world robustness.

## Download

```bash
uv run python -m src.data_pipeline.data_preparation street2shop --out-dir src/dataset/data/street2shop --max-pairs 5000
```

- `--max-pairs`: Limit pairs per split (default 5000 for quick setup)
- `--split`: train or test

Output: `street2shop/street/`, `street2shop/shop/`, `street2shop/pairs.csv`

## Joint Training

Enable in `configs/train.yaml` or use `configs/train_multitask.yaml`:

```yaml
use_street2shop: true
street2shop_weight: 0.5
street2shop_batch_size: 32
```

Then: `uv run python -m src.models.compatibility_trainer --config configs/train_multitask.yaml`
