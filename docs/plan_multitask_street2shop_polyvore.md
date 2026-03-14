# Plan: Multi-Task Training on Street2Shop + Merged Compatibility (CTL + Polyvore)

## Overview

Extend the training pipeline to two tasks:

1. **Outfit compatibility** (merged CTL + Polyvore) – same triplet objective from two data sources (Pinterest + Polyvore)
2. **Street2Shop** – street photo → shop product matching (real-world robustness)

Total loss: `compatibility_loss + α * street2shop_loss`.

## What Each Task Represents

| Task | Semantic meaning | Training signal |
|------|------------------|-----------------|
| **Outfit compatibility** (CTL + Polyvore) | "Complete the look" – items from the same outfit should be compatible; incompatible items should be far | Triplets: (anchor, pos, neg) where anchor+pos are compatible (same outfit or curated set), neg is incompatible. Merged from Pinterest CTL and Polyvore. |
| **Street2Shop** | "Find this exact product in the catalog" – a street/consumer photo should match the same shop product in embedding space | Triplets: (street_photo, matching_shop_photo, non_matching_shop_photo) |

**Combined effect**: Compatibility trains outfit matching from two sources (Pinterest + Polyvore); Street2Shop adds robustness to real-world street photos.

---

## 1. Street2Shop Integration

### 1.1 Data acquisition

- **Source**: http://www.tamaraberg.com/street2shop/wheretobuyit/
  - `photos.tar` – photo URLs
  - `meta.zip` – dataset triplets (street↔shop matches)
- **Alternative**: Hugging Face `petr7555/street2shop` if direct download is easier
- **Output layout**: `src/dataset/data/street2shop/`
  - `street/` – street photos
  - `shop/` – shop product photos
  - `pairs.csv` – street_path, shop_path, split

### 1.2 Download script

- **File**: `src/data_pipeline/data_preparation.py` (street2shop subcommand)
- **Steps**:
  1. Download `photos.tar` and `meta.zip` (or use Hugging Face `datasets`)
  2. Parse meta files to get street↔shop pairs
  3. Download images to `street/` and `shop/`
  4. Produce `pairs.csv` with columns: `street_path`, `shop_path`, `split`

### 1.3 Dataset and dataloader

- **File**: `src/schemas/dataset_schemas.py` (Street2ShopTripletDataset)
- **Classes**:
  - `Street2ShopPairDataset` – yields `(street_img, shop_img)` for contrastive training
  - `Street2ShopTripletDataset` – yields `(anchor_street, pos_shop, neg_shop)` for triplet loss
- **File**: `src/dataloader/data_loaders.py`
- **Class**: `Street2ShopDataloader` – `triplet_data_loader()` using same transforms as CTL

### 1.4 Config

- **File**: `configs/data_prep.yaml` – add `street2shop_root`, `street2shop_pairs_csv`
- **File**: `configs/train.yaml` – add `use_street2shop`, `street2shop_weight`, `street2shop_batch_size`

---

## 2. Merge Polyvore into Compatibility

### 2.1 Data acquisition

- **Source**: Hugging Face `owj0421/polyvore-outfits` or `Stylique/Polyvore`
  - Compatibility task: outfit items with compatible/incompatible labels
  - Format: `[label] [item_id_1] [item_id_2] ...` per line
- **Alternative**: GitHub `xthan/polyvore-dataset` – `compatibility_train.txt`, `compatibility_test.txt`
- **Output layout**: `src/dataset/data/polyvore/`
  - `images/` – item images (or symlinks)
  - `triplets.csv` – anchor_path, pos_path, neg_path, split

### 2.2 Download / prepare script

- **File**: `src/data_pipeline/data_preparation.py` (polyvore subcommand)
- **Steps**:
  1. Load from Hugging Face `datasets.load_dataset("owj0421/polyvore-outfits")` or download from GitHub
  2. Map item IDs to image paths
  3. Sample triplets: anchor + positive (compatible) + negative (incompatible or different outfit)
  4. Produce `triplets.csv` with `anchor_path`, `pos_path`, `neg_path`, `split`

### 2.3 Dataset and merge (no separate dataloader)

- **File**: `src/schemas/dataset_schemas.py` (PolyvoreTripletDataset)
- **Class**: `PolyvoreTripletDataset` – yields `(anchor, pos, neg)` for triplet loss (same format as CTL)
- **Merge**: Extend `FashionCompleteTheLookDataloader` (or add `UnifiedCompatibilityDataloader`) to accept optional Polyvore triplets path
- Use `ConcatDataset` to combine CTL and Polyvore when both are present; single compatibility loss

### 2.4 Config

- **File**: `configs/data_prep.yaml` – add `polyvore_root`, `polyvore_triplets_csv`
- **File**: `configs/train.yaml` – add `use_polyvore_in_compatibility` (default: true when Polyvore data exists)

---

## 3. Trainer Extension

### 3.1 Multi-task training loop

- **File**: `src/models/compatibility_trainer.py`
- **Changes**:
  - Add `use_street2shop`, `street2shop_weight`, `street2shop_batch_size` to `__init__`
  - Remove DeepFashion C2S support (replaced by Street2Shop; DeepFashion is password-protected)
  - In `train()`:
    - Single compatibility loader (CTL + Polyvore merged via ConcatDataset)
    - Create `Street2ShopDataloader` if `use_street2shop`
    - Use `itertools.cycle` for Street2Shop loader
    - Each step: `compatibility_loss + α * street2shop_loss`
    - Log each loss component separately

### 3.2 Config entry point

- **File**: `configs/train_multitask.yaml`
- **Contents**:
  ```yaml
  use_street2shop: true
  street2shop_weight: 0.5
  street2shop_batch_size: 32
  use_polyvore_in_compatibility: true
  ```

---

## 4. Constants and Paths

- **File**: `src/constants.py`
  - Add `STREET2SHOP_ROOT`, `POLYVORE_ROOT`
- **File**: `src/config/config.py`
  - Re-export new constants

---

## 5. Documentation

- **File**: `docs/street2shop.md` – download, structure, usage
- **File**: `docs/polyvore.md` – download, structure, usage
- **File**: `README.md` – add section on multi-task training with Street2Shop + Polyvore

---

## 6. Evaluation (optional)

- **Street2Shop**: Add `evaluate_street2shop_recall()` in `evaluation.py` (recall@k on street→shop retrieval)

---

## 7. Remove DeepFashion C2S Code

DeepFashion is password-protected and inaccessible; Street2Shop replaces it for street→shop robustness. Remove all DeepFashion-related code:

- **Delete**: `src/dataset/deepfashion_c2s.py`
- **Delete**: `src/dataset/data/prepare_deepfashion_c2s.py`
- **Delete**: `src/dataset/data/download_deepfashion_c2s.py`
- **Delete**: `configs/train_deepfashion.yaml`
- **Delete**: `docs/deepfashion.md`
- **Edit** `src/dataloader/data_loaders.py`: Remove `DeepFashionC2SDataloader` class and `DeepFashionC2STripletDataset` import
- **Edit** `src/models/compatibility_trainer.py`: Remove `use_deepfashion_c2s`, `deepfashion_c2s_weight`, `deepfashion_c2s_batch_size`, `DeepFashionC2SDataloader` import and all related logic
- **Edit** `src/models/evaluation.py`: Remove `evaluate_deepfashion_c2s_recall()` and its call in `__main__`
- **Edit** `configs/train.yaml`: Remove `use_deepfashion_c2s`, `deepfashion_c2s_weight`, `deepfashion_c2s_batch_size`
- **Edit** `configs/data_prep.yaml`: Remove `deepfashion_c2s_root`, `deepfashion_c2s_pairs_csv`
- **Edit** `src/constants.py`: Remove `DEEPFASHION_C2S_ROOT`
- **Edit** `src/config/config.py`: Remove `DEEPFASHION_C2S_ROOT` re-export
- **Edit** `README.md`: Remove DeepFashion section and joint training instructions
- **Edit** `.gitignore`: Remove `src/dataset/data/deepfashion_c2s/`; add `street2shop/`, `polyvore/`

---

## Implementation Order

0. **Prerequisites**: CTL (fashion_v2) is required for the compatibility task. Run `prepare_stl_ctl --ctl-train --ctl-test --data-dir src/dataset/data`.
1. **Remove DeepFashion**: Delete files and strip DeepFashion code from trainer, config, evaluate, Dataloader, constants, README
2. Street2Shop: download script → dataset → dataloader → config
3. Polyvore: prepare script → dataset; merge into compatibility dataloader (ConcatDataset)
4. Trainer: extend for Street2Shop only (compatibility already handles CTL+Polyvore)
5. Docs and evaluation hooks
