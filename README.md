# Complete the Look Recommendation System

Building a comprehensive recommendation system prototype for online ecommerce. Please kindly provide proper citation if you plan to use any part of the project.

## Introduction

With the accelerated online eCommerce scene driven by the contactless shopping style in recent years, having a great recommendation system is essential to the business' success. However, it has always been challenging to provide any meaningful recommendations with the absence of user interaction history, known as the cold start problem.  In this project, we attempted to create a comprehensive recommendation system that recommends both similar and complementary products using the power of deep learning and visual embeddings, which would effectively recommend products without need any knowledge of user preferences, user history, item propensity, or any other data.


## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if needed): curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Activate the virtual environment: `source .venv/bin/activate` (or use `uv run` which auto-uses it).

## Datasets

| Dataset | Purpose |
|---------|---------|
| [Complete the Look (CTL)](https://github.com/eileenforwhat/complete-the-look-dataset) | Outfit compatibility (anchor/pos/neg triplets). Downloaded as `fashion_v2`. |
| [Shop the Look (STL)](https://github.com/kang205/STL-Dataset) | Similar product recommendations (alternative to CTL). |
| [Street2Shop](docs/street2shop.md) | Street photo → shop product matching (multi-task). Fetched from Hugging Face. |
| [Polyvore](docs/polyvore.md) | Additional outfit compatibility data, merged with CTL (multi-task). Use `--download-images` or add manually. |

**STL/CTL setup**: Clone both repos into `src/dataset/data/`:
- [STL-Dataset](https://github.com/kang205/STL-Dataset) for `fashion.json` (STL)
- [complete-the-look-dataset](https://github.com/eileenforwhat/complete-the-look-dataset) for `raw_train.tsv` and `raw_test.tsv` (CTL)

The preparation script fetches images from these metadata files.

## Quick Run Instructions

#### Recommend Similar Products
1. Download data: `uv run python -m src.data_pipeline.data_preparation stl_ctl --ctl-test --data-dir src/dataset/data` (or `--stl` for STL)
2. Get similar product embedding: `uv run python -m src.features.embeddings` (2+ hours without GPU)
3. Recommend similar products: `uv run python -m src.recommend_cli`
4. Streamlit UI: `uv run streamlit run streamlit_app.py`

#### Recommend Compatible Products
1. Download data: `uv run python -m src.data_pipeline.data_preparation stl_ctl --ctl-train --ctl-test --data-dir src/dataset/data`
2. Train compatible model: `uv run python -m src.models.compatibility_trainer`
3. Get compatible product embedding: `uv run python -m src.features.embeddings` (see `__main__` in embeddings.py)
4. Evaluate: `uv run python -m src.models.evaluation`
5. Recommend compatible products: `uv run python -m src.recommend_cli` (select `recommend_complementary_products` in `__main__`)

#### Multi-task training (CTL + Street2Shop + Polyvore)

Trains on three data sources: CTL provides the compatibility base; Street2Shop adds street-to-shop robustness; Polyvore augments compatibility triplets.

1. Download CTL (fashion_v2): `uv run python -m src.data_pipeline.data_preparation stl_ctl --ctl-train --ctl-test --data-dir src/dataset/data`
2. Download Street2Shop: `uv run python -m src.data_pipeline.data_preparation street2shop --out-dir src/dataset/data/street2shop`
3. Prepare Polyvore: `uv run python -m src.data_pipeline.data_preparation polyvore --out-dir src/dataset/data/polyvore` (use `--download-images` to fetch images from Hugging Face, or add manually)
4. Train: `uv run python -m src.models.compatibility_trainer --config configs/train_multitask.yaml`

See [docs/street2shop.md](docs/street2shop.md) and [docs/polyvore.md](docs/polyvore.md) for details.

## Results
Samples of similar product recommendation
(on the left is the query product, on the right is the top 5 recommended similar products)

<img width="750" alt="image" src="https://user-images.githubusercontent.com/18410378/146464288-d3960443-a616-49fc-b250-eadb1f751927.png">


Samples of compatible product recommendation
(on the left is the query product, on the right is the top 5 recommended compatible products)

<img width="750" alt="image" src="https://user-images.githubusercontent.com/18410378/146464340-db108af4-ae66-409b-97fc-cb993c7a17bb.png">
