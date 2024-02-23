# SVM for Article Popularity Prediction

This project demonstrates the application of Support Vector Machine (SVM) models for predicting the popularity of articles submitted to a journal. Utilizing features extracted from articles, the goal is to classify articles as either popular or not popular for publication based on their characteristics. The project explores different SVM kernels, including linear and polynomial, to find the best performing model.

## Project Overview

- **Data Source**: The dataset comprises features extracted from articles, labeled with their popularity as either high or low. The dataset is stored in a CSV file, and supplementary information about each feature is provided in a Notepad file named "information".

- **Goal**: Implement an SVM model to classify articles based on their extracted features into popular or not popular categories for publication.

- **Preprocessing**:
  - The dataset is loaded from a CSV file.
  - Categorical target variables are encoded into numeric format.
  - Features are scaled to standardize the data.

- **Model Training**:
  - The dataset is split into training and testing sets.
  - SVM models with linear and polynomial kernels are trained on the dataset.
  - Model performance is evaluated using accuracy and confusion matrices.

- **Model Selection**:
  - Cross-validation is performed to find the best value for the regularization parameter C.
  - The best-performing model is identified based on cross-validation scores.

- **Model Deployment**:
  - The selected SVM model is saved using `pickle` for future use.

## Key Components

- **Data Preprocessing**: Utilizes `sklearn`'s `scale` function for feature scaling and `LabelEncoder` for encoding target labels.
- **SVM Models**: Explores linear and polynomial kernels to classify articles. The `sklearn.svm.SVC` class is used for model implementation.
- **Model Evaluation**: Accuracy scores and confusion matrices are used to assess model performance.
- **Cross-Validation**: Utilizes `cross_val_score` for selecting the best regularization parameter C.
- **Model Saving and Loading**: Implements `pickle` for saving and loading the trained SVM model.

## Usage

1. **Environment Setup**: Ensure `numpy`, `pandas`, `sklearn`, and `pickle` are installed in your Python environment.
2. **Data Preparation**: Load your dataset, preprocess the features, and split the data into training and testing sets.
3. **Model Training and Selection**:
   - Train SVM models with different kernels and parameters.
   - Perform cross-validation to select the best model.
4. **Evaluation**: Assess the selected model's performance on the test set.
5. **Deployment**: Save the trained model using `pickle` for future predictions.

## Model Performance

The project concludes with the evaluation of SVM models on the test dataset. The linear SVM model, after cross-validation for parameter selection, is chosen based on its performance metrics. The final model is saved for future use, allowing for the quick classification of new articles based on their popularity.

This example showcases the practical application of SVM models in classifying articles for publication based on extracted features, providing a foundation for further exploration and model optimization.
