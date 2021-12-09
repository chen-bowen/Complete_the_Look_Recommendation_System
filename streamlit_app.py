import streamlit as st

from src.recommend import recommend_similar_products
from src.utils.image_utils import display_recommended_products


def convert_to_url(signature):
    """convert image"""
    prefix = "http://i.pinimg.com/400x/%s/%s/%s/%s"
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)


def handle_input(input: int):
    hi = recommend_similar_products(product_id=input)
    fig = display_recommended_products(
        hi["input_product"]["image_path"],
        *[rec["image_path"] for rec in hi["recommended_products"]],
        [round(rec["similarity_score"], 3) for rec in hi["recommended_products"]],
        save_image=False
    )
    return fig

st.title('Complete The Look Project')
product_id_input = st.number_input("Product ID", min_value=0, max_value=38109, value=140)
col1, col2 = st.columns(2)
submit_btn = col1.button("Submit")
ballon_btn = col2.button("What's This?")
if submit_btn:
    fig = handle_input(product_id_input)
    with st.container():
        st.pyplot(fig)

if ballon_btn:
  st.balloons()