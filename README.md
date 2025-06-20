# Soil Type Classification and Crop Recommendation

## Project Overview

This project combines the power of machine learning and data analysis to create a system that:

- Classifies soil types using a trained model with high accuracy.
- Recommends suitable crops based on the classified soil type and its nutrient requirements.

By combining two datasets from Kaggle, this project provides a comprehensive solution for farmers and agricultural researchers to make data-driven decisions for optimal crop cultivation.

## Features

- Soil Type Classification:
  - Built a machine learning model to classify different soil types with 98% accuracy on both training and testing datasets.

- Crop Recommendation:
  - Created a nutrient-soil mapping table.
  - Recommends crops based on the nutrient composition of the classified soil type.
 
## Workflow

1. Dataset Preparation
  - Datasets Used:
    - Combined two Kaggle datasets to expand the range of soil types.
    - Cleaned and preprocessed the data to ensure quality and consistency.
  - Datasets Links : 
    - https://www.kaggle.com/datasets/jayaprakashpondy/soil-image-dataset
    - https://www.kaggle.com/datasets/prasanshasatpathy/soil-types

2. Model Training

  - Used a machine learning model to classify soil types.
    - Achieved an impressive accuracy of 98% for both training and testing datasets.

3. Nutrient-Soil Mapping

  - Designed a mapping table that links different soil types with their respective nutrient compositions.

4. Crop Recommendation System
  - Utilized the nutrient-soil mapping to recommend crops best suited for each soil type.

5. Web Interface

  - Built a user-friendly web application using Flask.

  - Designed intuitive HTML/CSS templates to enable users to:

    - Input soil data.

    - View the classified soil type.

    - Get crop recommendations instantly.


## Installation

1. Clone this repository:
 - git clone https://github.com/yourusername/soil-crop-recommendation.git

2. Navigate to the project directory:
  - cd soil-crop-recommendation

3. Install the required libraries:
   - pip install -r requirements.txt
  
## Running the Project
- Train the model (optional, pre-trained model included) : python Training/soil-type-classification.ipynb
- Start the web application : python app.py
  
## Results

- Model Accuracy: 98% on both training and testing datasets.
- Crop Recommendations: Accurate recommendations based on nutrient-soil mapping.
#   F a r m - T e c h  
 