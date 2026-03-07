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

The dataset used is the [shop the look dataset](https://github.com/kang205/STL-Dataset) and the [complete the look dataset](https://github.com/eileenforwhat/complete-the-look-dataset) from Pinterest. Thank you for kindly sharing these great data sources to make this project possible.

## Quick Run Instructions

#### Recommend Similar Products
1. Download data: `uv run python -m src.dataset.data.download_data --ctl-test` (or `--stl` for STL)
2. Get similar product embedding: `uv run python -m src.features.Embedding` (2+ hours without GPU)
3. Recommend similar products: `uv run python -m src.recommend`
4. Streamlit UI: `uv run streamlit run streamlit_app.py`

#### Recommend Compatible Products
1. Download data: `uv run python -m src.dataset.data.download_data --ctl-train --ctl-test`
2. Train compatible model: `uv run python -m src.models.training`
3. Get compatible product embedding: `uv run python -m src.features.Embedding` (see `__main__` in Embedding.py)
4. Evaluate: `uv run python -m src.models.evaluate`
5. Recommend compatible products: `uv run python -m src.recommend` (select `recommend_complementary_products` in `__main__`)

## Results
Samples of similar product recommendation
(on the left is the query product, on the right is the top 5 recommended similar products)

<img width="750" alt="image" src="https://user-images.githubusercontent.com/18410378/146464288-d3960443-a616-49fc-b250-eadb1f751927.png">


Samples of compatible product recommendation
(on the left is the query product, on the right is the top 5 recommended compatible products)

<img width="750" alt="image" src="https://user-images.githubusercontent.com/18410378/146464340-db108af4-ae66-409b-97fc-cb993c7a17bb.png">



