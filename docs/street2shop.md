# Street2Shop Dataset

Street2Shop provides street photo to shop product matching for real-world robustness.

## Download

```bash
uv run python -m src.data_pipeline.data_preparation street2shop
```

Options in `configs/data_prep.yaml` under `street2shop`:
- `out_dir`, `max_pairs`, `split`, `streaming`

Output: `street2shop/street/`, `street2shop/shop/`, `street2shop/pairs.csv`

## Joint Training

Enable in `configs/train.yaml` or use `configs/train_multitask.yaml`:

```yaml
use_street2shop: true
street2shop_weight: 0.5
street2shop_batch_size: 32
```

Then: `uv run python -m src.models.compatibility_trainer --config configs/train_multitask.yaml`
