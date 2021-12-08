import streamlit as st
from src.models.recommend import recommend_similar_products
from src.utils.show_images import print_image

def convert_to_url(signature):
    """convert image"""
    prefix = "http://i.pinimg.com/400x/%s/%s/%s/%s"
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)

def handle_input(input: int):
  hi = recommend_similar_products(product_id=input)
  fig = print_image(
        hi["input_product"]["image_path"],
        *[rec["image_path"] for rec in hi["recommended_products"]],
        [round(rec["similarity_score"], 3) for rec in hi["recommended_products"]],
        save_image=False
    )
  return fig
product_id_input = st.number_input('Product ID', min_value=0, max_value=38109, value=5559)
submit_btn = st.button('Submit')
if(submit_btn):
  fig = handle_input(heproduct_id_inputllo)
  with st.container():
    st.pyplot(fig)