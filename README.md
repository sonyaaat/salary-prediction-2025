# Predicting Salary Expectations Using Machine Learning Methods  
**Thesis Work Experiments**

This repository contains all experimental notebooks and scripts used for the Bachelor's thesis project on predicting salary expectations in the IT sector using machine learning methods.

## Project Structure

### 1. `data_merging.ipynb`
  Merges 3 initial datasets into a single dataset. Ensures consistent structure and format for further analysis.

### 2. `data_preprocessing.ipynb`
  Handles all preprocessing steps, such as:
  - Removing duplicates and outliers
  - Encoding categorical variables
  - Filling in or removing missing values
  - Normalization and feature selection
  - Columns mapping

### 3. `exploratory_data_analysis.ipynb`  
  Performs exploratory data analysis (EDA) using visualizations and descriptive statistics to:
  - Understand data distribution
  - Discover correlations between features
  - Identify trends and insights related to salary expectations

### 4. `models_training_results.ipynb`  
  Trains and evaluates machine learning models s
  Includes model comparison using metrics such as MAE, RMSE, and R² score.

### 5. `initial_data`  folder 
  Contains three raw datasets obtained from DOU.ua. These are the original sources used in the data_merging.ipynb notebook to build a unified dataset for analysis and modeling.

### 6. `utils`  folder
 Contains the merged raw dataset  and cleaned and preprocessed dataset.

### 7. `processed_data`  folder
 Contains helper tools used during model development, including a preprocessing script (preprocessing.py), a saved scaler object for feature normalization (scale.pkl), and a target encoder used for transforming categorical features (target_encoder.joblib).
## Data Sources

The main dataset used in this project is provided by **DOU.ua**, one of the largest Ukrainian IT communities. It contains self-reported salary survey data from Ukrainian tech professionals.

The original dataset is publicly available and can be accessed via the following GitHub repository:  
🔗 [https://github.com/devua/csv/tree/master/salaries](https://github.com/devua/csv/tree/master/salaries)

## Author
**Sofiia Tkach**  
Bachelor’s Thesis, Group CS-416  
Department of Artificial Intelligence Systems
Year: 2025