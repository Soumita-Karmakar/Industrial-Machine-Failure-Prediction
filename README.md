# Industrial Machine Failure Prediction

## 📌 Project Overview

The **Industrial Machine Failure Prediction** project uses Data Science and Machine Learning techniques to analyse industrial machine operating data and predict potential machine failures.

The project uses the AI4I 2020 Predictive Maintenance Dataset to investigate the relationship between machine operating conditions, sensor measurements, and machine failures.

The main objective is to develop a classification-based prediction system that can identify whether an industrial machine is likely to experience failure based on its operating measurements.

This project demonstrates how Machine Learning can support predictive maintenance by helping identify potential machine failures and reduce unexpected equipment downtime.

## 🎯 Project Objectives

- Analyse industrial machine operating data.
- Perform data cleaning and Exploratory Data Analysis (EDA).
- Identify patterns and relationships between sensor measurements and machine failures.
- Train multiple Machine Learning classification models.
- Compare model performance using appropriate evaluation metrics.
- Perform cross-validation to assess model performance.
- Apply hyperparameter tuning to improve classification performance.
- Select the best-performing model.
- Predict machine failure using new, unseen machine data.
- Save and reuse the trained model without retraining.

## 📂 Project Structure

```text
Industrial-Machine-Failure-Prediction/
│
├── dataset/
│   └── ai4i2020.csv
│
├── models/
│   └── machine_failure_model.pkl
│
├── notebooks/
│   └── industrial_machine_failure_prediction.ipynb
│
└── README.md
```

## 📊 Dataset Description

**Dataset Name:** AI4I 2020 Predictive Maintenance Dataset

**Dataset Source:** UCI Machine Learning Repository

The dataset contains **10,000 records and 14 original columns** representing industrial machine operating conditions and failure information.

### Dataset Features

| Feature | Description |
|---|---|
| UDI | Unique data identifier |
| Product ID | Machine product identifier |
| Type | Machine quality type (L, M, H) |
| Air temperature [K] | Air temperature |
| Process temperature [K] | Process temperature |
| Rotational speed [rpm] | Machine rotational speed |
| Torque [Nm] | Machine torque |
| Tool wear [min] | Tool wear time |
| Machine failure | Indicates whether machine failure occurred |
| TWF | Tool Wear Failure indicator |
| HDF | Heat Dissipation Failure indicator |
| PWF | Power Failure indicator |
| OSF | Overstrain Failure indicator |
| RNF | Random Failure indicator |

**Target Variable:** `Machine failure`

- `0` – No Machine Failure
- `1` – Machine Failure

The target variable is imbalanced, with 9,661 records representing no failure and 339 records representing machine failure.

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

- Checked for missing values and duplicate records.
- Removed unnecessary identifier columns such as UDI and Product ID.
- Excluded individual failure indicator columns (TWF, HDF, PWF, OSF, RNF) to avoid target leakage.
- Encoded the categorical machine Type feature using One-Hot Encoding.
- Separated input features and the target variable.
- Split the dataset into training and testing sets using stratified sampling.
- Applied StandardScaler to numerical sensor features for Logistic Regression.

The dataset was divided into 80% training data and 20% testing data.

## 🤖 Machine Learning Algorithms

Three Machine Learning classification algorithms were implemented.

### 1. Logistic Regression

Used as a baseline classification model to predict machine failure based on sensor measurements.

Feature scaling and class weighting were applied to handle differences in feature scales and class imbalance.

### 2. Decision Tree Classifier

Used to identify machine failure patterns through a tree-based classification approach.

Class weighting and hyperparameter tuning were applied to improve its performance.

### 3. Random Forest Classifier

Used as an ensemble learning method that combines multiple decision trees to predict machine failure.

It was tuned using GridSearchCV and selected as the final model based on its cross-validation F1-score.

## 📈 Model Evaluation Results

The models were evaluated using accuracy, precision, recall, and F1-score.

### Initial Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.45% | 14.18% | 82.35% | 24.19% |
| Decision Tree | 97.80% | 70.69% | 60.29% | 65.08% |
| Random Forest | 97.90% | 96.43% | 39.71% | 56.25% |

### Cross-Validation Results

Stratified 5-fold cross-validation was performed using the training dataset.

| Model | Mean Accuracy | Mean Precision | Mean Recall | Mean F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.25% | 13.74% | 80.44% | 23.46% |
| Decision Tree | 97.64% | 68.97% | 56.10% | 61.46% |
| Random Forest | 97.71% | 89.94% | 36.55% | 51.93% |

### Hyperparameter Tuning Results

GridSearchCV was applied to tune all three classification models using F1-score as the scoring metric.

| Model | Best Cross-Validation F1-score |
|---|---:|
| Logistic Regression | 0.2346 |
| Decision Tree | 0.6395 |
| Random Forest | **0.6419** |

### Selected Model: Random Forest Classifier

The Random Forest Classifier was selected because it achieved the highest cross-validation F1-score among the tuned models.

Its best hyperparameters were:

```python
{
    'max_depth': 20,
    'min_samples_leaf': 2,
    'min_samples_split': 5,
    'n_estimators': 200
}
```

## 🎯 Final Model Performance

The selected tuned Random Forest model was evaluated on the untouched test dataset.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 98% |
| Precision (Failure Class) | 82% |
| Recall (Failure Class) | 62% |
| F1-score (Failure Class) | 71% |

### Confusion Matrix

```text
                 Predicted
                 No Failure  Failure

Actual No Failure    1923       9
Actual Failure         26      42
```

The model correctly identified 42 out of 68 actual machine failures and produced 9 false-positive predictions.

**Note:** Although the model achieved high accuracy, its failure-class recall indicates that some actual machine failures are still missed. The classification results should therefore be interpreted carefully, particularly because the dataset is highly imbalanced.

## 📉 Data Visualization and Analysis

The project includes visualizations for:

- Machine failure distribution.
- Machine type distribution.
- Machine failure rates across different machine types.
- Sensor measurement distributions.
- Boxplots comparing sensor measurements for failure and non-failure records.
- Feature correlation heatmap.
- Model performance comparison.
- Confusion matrix of the selected model.

## 🔍 Hyperparameter Optimization

GridSearchCV was used to identify suitable hyperparameters for:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Stratified 5-fold cross-validation was used during hyperparameter tuning.

The F1-score was selected as the optimization metric because machine failure records represent only a small proportion of the dataset.

## 💻 New Machine Failure Prediction

The project includes an interactive command-line prediction system that accepts new machine operating measurements.

### User Input Features

- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]
- Machine Type (L, M, H)

### Prediction Output

The system provides:

- Machine Failure: Yes or No.
- Estimated Machine Failure Probability.

### Sample Input

| Sensor Measurement | Example Value |
|---|---:|
| Air Temperature | 300 K |
| Process Temperature | 310 K |
| Rotational Speed | 1500 rpm |
| Torque | 45 Nm |
| Tool Wear | 120 min |
| Machine Type | M |

### Example Output

```text
===== MACHINE FAILURE PREDICTION =====

Result: MACHINE FAILURE - NO

Estimated Failure Probability: 0.0 %
```

*Illustrative example only. Actual predictions depend on the trained model and the entered machine measurements.*

## 💾 Model Saving and Loading

The selected Random Forest model was saved using the Joblib library.

The saved file contains:

- Trained Random Forest model.
- Expected input feature names and order.

The saved model can be loaded and used to predict new machine records without retraining.

**Saved Model:** `machine_failure_model.pkl`

## 🛠️ Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Machine Learning
- Data Science

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Soumita-Karmakar/Industrial-Machine-Failure-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd Industrial-Machine-Failure-Prediction
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 4. Open the Notebook

Open the notebook using Google Colab or Jupyter Notebook.

### 5. Load the Dataset

Ensure the dataset is available in the `dataset` folder and update the dataset path in the notebook if necessary.

### 6. Run the Notebook

Execute the notebook cells sequentially to perform:

- Data analysis and preprocessing.
- Model training.
- Model evaluation.
- Cross-validation.
- Hyperparameter tuning.
- Best model selection.
- New machine failure prediction.

### 7. Load the Saved Model

The saved model can be loaded using:

```python
import joblib

model_data = joblib.load("models/machine_failure_model.pkl")

loaded_model = model_data["model"]
feature_columns = model_data["feature_columns"]
```

## 🚀 Future Improvements

- Improve recall for the machine failure class.
- Explore advanced class imbalance handling techniques.
- Experiment with additional classification algorithms.
- Perform further feature engineering.
- Explore probability threshold optimization.
- Develop a web-based machine failure prediction interface.
- Deploy the trained model for practical predictive maintenance applications.

## 👩‍💻 Author

**Soumita Karmakar**

MCA Student | Brainware University

GitHub: [Soumita-Karmakar](https://github.com/Soumita-Karmakar)

