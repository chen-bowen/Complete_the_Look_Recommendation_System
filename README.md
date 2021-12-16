# Complete the Look Recommendation System

## Introduction

With the accelerated online ecommerce scene driven by the contactless shopping style in recent years, having a great recommendation system is essential to the business' success. However, it has always been challenging to provide any meaningful recommendations with the absence of user interaction history, known as the cold start problem. Our motivation is to create a cold start recommendation engine that does not need any knowledge of user preferences, user history, item propensity or any other data to recommend products to the customer. In this project, we attempted to create a comprehensive recommendation system that recommends both similar and complementary products using the power of deep learning, which would resolve the issue for lacking of user history.


## Datasets

The dataset used is the [shop the look dataset](https://github.com/kang205/STL-Dataset) and the [complete the look dataset](https://github.com/eileenforwhat/complete-the-look-dataset) from Pinterest. Thank you for kindly sharing these great data sources to make this project possible.

## Quick Run Instructions

#### Recommend Similar Products
1. Download data - Run `cd src/dataset/data` and run `python download_data.py`
2. Get similar product embedding - Run `cd src`, make sure the in the `features/Embedding.py`, the class method `similar_product_embedding` is being selected, then run ` PYTHONPATH=../:. python features/Embedding.py` (be careful this could take up to 2 hours without the presence of a GPU)
3. Recommend Similar Product - Run `cd src`, make sure the in the `recommend.py`, the function `recommend_similar_products` is being selected, then run ` PYTHONPATH=../:. python recommend.py`

#### Recommend Compatible Products
1. Download data - Run `cd src/dataset/data` and run `python download_data.py`
2. Train compatible model - Run `cd src` and run ` PYTHONPATH=../:. python models/training.py`
3. Get compatible product embedding - Run `cd src`, make sure the in the `features/Embedding.py`, the class method `similar_product_embedding` is being selected, then run ` PYTHONPATH=../:. python features/Embedding.py` (be careful this could take up to 15 hours without the presence of a GPU, 7 hours with GPU)<img 
4. Evaluate the compatible model -  Run `cd src` and run ` PYTHONPATH=../:. python models/evaluate.py`
5. Recommend Compatible Product - Run `cd src`, make sure the in the `recommend.py`, the function `recommend_compatible_products` is being selected, then run ` PYTHONPATH=../:. python recommend.py`

## Results
Samples of similar product recommendation

<img width="500" alt="image" src="https://user-images.githubusercontent.com/18410378/146464288-d3960443-a616-49fc-b250-eadb1f751927.png">


Samples of compatible product recommendation


<img width="500" alt="image" src="https://user-images.githubusercontent.com/18410378/146464340-db108af4-ae66-409b-97fc-cb993c7a17bb.png">



