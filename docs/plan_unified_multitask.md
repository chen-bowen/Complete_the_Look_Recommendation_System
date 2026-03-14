# Unified Plan: Multi-Task Training (CTL + Street2Shop + Polyvore)

## Overview

Three tasks with shared CompatibilityModel:

1. **Outfit compatibility** (CTL + Polyvore merged) – triplet loss
2. **Street2Shop** – street photo to shop product retrieval
3. **FashionIQ** (future) – text-grounded composed retrieval

Total loss: `compatibility_loss + alpha * street2shop_loss` (+ beta * fashioniq_loss when added).

## What Each Task Represents

| Task | Semantic meaning | Training signal |
|------|------------------|-----------------|
| **Outfit compatibility** (CTL + Polyvore) | "Complete the look" – compatible outfit items close, incompatible far | Triplets: (anchor, pos, neg) |
| **Street2Shop** | "Find this exact product" – street photo matches shop product | Triplets: (street, pos_shop, neg_shop) |
| **FashionIQ** (future) | "Retrieve by image + text" – e.g. "more casual" | (ref+text, candidate, neg) |

## Implementation Status

- [x] Remove DeepFashion C2S
- [x] Street2Shop: download, dataset, dataloader, trainer
- [x] Polyvore: prepare, dataset, merge into compatibility
- [x] Trainer extension, configs, docs
- [ ] FashionIQ: text encoder, fusion, dataloader (future)

## Files Added/Changed

- `src/data_pipeline/data_preparation.py` (stl_ctl, street2shop, polyvore subcommands)
- `src/schemas/dataset_schemas.py` – Street2ShopTripletDataset, PolyvoreTripletDataset, STL/CTL datasets
- `src/dataloader/data_loaders.py` – Street2ShopDataloader, Polyvore merge
- `src/models/compatibility_trainer.py` – Street2Shop + Polyvore support
- `configs/train_multitask.yaml`
- `docs/street2shop.md`, `docs/polyvore.md`
