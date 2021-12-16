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

<img width="421" alt="image" src="https://user-images.githubusercontent.com/18410378/146463805-778031ea-61c8-45dc-aa31-5e452cf891ee.png">
<img width="421" alt="image" src="https://user-images.githubusercontent.com/18410378/146463816-877cb0f7-1d68-445f-adde-b47047f2743b.png">
<img width="427" alt="image" src="https://user-images.githubusercontent.com/18410378/146463828-606311f5-2432-41c4-b208-53351644215d.png">
<img width="427" alt="image" src="https://user-images.githubusercontent.com/18410378/146463834-00b73b3e-0350-411e-9c8c-074cc60feb8c.png">
<img width="427" alt="image" src="https://user-images.githubusercontent.com/18410378/146463840-805759a3-3859-4821-806b-ec014f351630.png">
<img width="427" alt="image" src="https://user-images.githubusercontent.com/18410378/146463849-88f06c87-2309-453a-8d34-fdeaaba93b78.png">

Samples of compatible product recommendation

<img width="421" alt="image" src="https://user-images.githubusercontent.com/18410378/146463893-cc555b4d-e11b-4940-bfe1-32accc1db13f.png">
<img width="402" alt="image" src="https://user-images.githubusercontent.com/18410378/146463909-b3371f7a-f5c1-4e6f-8693-346746beb8dd.png">
<img width="403" alt="image" src="https://user-images.githubusercontent.com/18410378/146463921-6fa8a5d1-4ce1-46e5-a143-a607507a94ec.png">
<img width="394" alt="image" src="https://user-images.githubusercontent.com/18410378/146463929-63b965b6-af1d-4299-bcb1-477afa5e3a5b.png">
<img width="402" alt="image" src="https://user-images.githubusercontent.com/18410378/146463990-d66008c0-4859-4ee3-a608-9c18855c42d9.png">
<img width="402" alt="image" src="https://user-images.githubusercontent.com/18410378/146464066-0633e173-9715-44e8-8d39-a1936b28d238.png">





