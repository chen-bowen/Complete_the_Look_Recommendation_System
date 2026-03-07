"""Streamlit UI for similar product recommendations."""

import streamlit as st

from src.inference.recommender import SimilarProductRecommender
from src.utils.image_utils import display_recommended_products

# Instantiate recommender once at startup
recommender = SimilarProductRecommender(task_name="similar_product")


def handle_input(product_id: int):
    """Get recommendations and return figure for display."""
    result = recommender.recommend(product_id=product_id, top_n=5)
    fig = display_recommended_products(
        result["input_product"]["image_path"],
        *[rec["image_path"] for rec in result["recommended_products"]],
        [round(rec["similarity_score"], 3) for rec in result["recommended_products"]],
        save_image=False,
    )
    return fig


st.title("Complete The Look Project")
product_id_input = st.number_input(
    "Product ID", min_value=0, max_value=38109, value=140
)
col1, col2 = st.columns(2)
submit_btn = col1.button("Submit")
ballon_btn = col2.button("What's This?")

if submit_btn:
    fig = handle_input(product_id_input)
    with st.container():
        st.pyplot(fig)

if ballon_btn:
    st.balloons()
