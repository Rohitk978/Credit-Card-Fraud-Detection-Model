# Credit Card Fraud Detection

## Overview
This project develops a machine learning model to detect fraudulent credit card transactions using a neural network. It analyzes transaction data to predict whether a transaction is fraudulent or legitimate, addressing the challenge of highly imbalanced data where fraud cases are rare. The project includes data preprocessing, feature engineering, model training, and performance evaluation, with a focus on maximizing fraud detection while minimizing false positives.

## Libraries
- **NumPy**: For numerical computations and array handling.
- **Pandas**: For data manipulation and CSV handling.
- **Scikit-learn**: For machine learning utilities and evaluation metrics.
- **TensorFlow** or **PyTorch**: For building and training the neural network.
- **Matplotlib**: For visualization of performance metrics and feature importance.
- **Seaborn**: For enhanced data visualization (optional, included in requirements).

## Project Logic Highlights
- **Data Loading**: Imports and validates credit card transaction data from CSV files.
- **Data Cleaning**: Scales numerical features and encodes categorical data for model compatibility.
- **Feature Creation**: Generates new features like geographical distances and time-based trends to improve detection.
- **Model Design**: Implements a neural network tailored for binary classification with imbalanced data handling.
- **Training Optimization**: Applies techniques such as oversampling or class weighting to address imbalance.
- **Performance Assessment**: Evaluates using ROC AUC, precision, recall, and confusion matrix for robust fraud detection.
- **Interpretability Analysis**: Identifies key features driving fraud predictions for better understanding.

## Dataset
- **Format**: Credit card transaction data in CSV format, with columns for transaction details (e.g., amount, time, location) and a binary label (0 for legitimate, 1 for fraudulent).
- **Example**:
  ```
  amount,time,location,label
  125.50,1630459200,New York,0
  23.75,1630459250,Los Angeles,1
  ```
- **Source**: A suitable dataset is the Kaggle Credit Card Fraud Detection dataset: **[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** (not included, users must download and place in the project directory).

## Key Features
- **Data Preprocessing**: Cleans and transforms transaction data with numerical scaling and categorical encoding.
- **Feature Engineering**: Creates transaction pattern features like geographical distances and temporal trends.
- **Model**: Neural network optimized for binary classification of imbalanced data.
- **Training**: Incorporates imbalance handling techniques for effective fraud detection.
- **Evaluation**: Uses ROC AUC, precision, recall, and confusion matrix for comprehensive performance assessment.
- **Interpretability**: Analyzes feature importance to highlight key fraud indicators.

## Installation
To set up the project locally, follow these steps:
- **Clone the Repository**:
  ```
  git clone https://github.com/your-username/credit-card-fraud-detection.git
  cd credit-card-fraud-detection
  ```
- **Create a Virtual Environment** (recommended):
  ```
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
- **Install Dependencies**:
  ```
  pip install -r requirements.txt
  ```
- **Place Dataset**: Add your dataset (e.g., `train_data.csv`, `test_data.csv`) in the project directory.

## Usage
- **Update File Paths**: Modify file paths in `fraud_detection.py` to match your dataset location.
- **Run the Script**:
  ```
  python fraud_detection.py
  ```
  - **Process**: Preprocesses data, trains the neural network, evaluates performance, generates visualizations, and saves predictions.
  - **Output**: Displays metrics (ROC AUC, precision, recall, F1-score), saves visualizations in `figures/`, and exports predictions to `predictions.csv`.

## Methodology
### Data Preprocessing
- Scales numerical transaction data (e.g., amounts, times) for consistency.
- Encodes categorical variables (e.g., locations) for model input.

### Feature Engineering
- Computes geographical distances between transaction locations.
- Extracts temporal trends, such as time of day or frequency patterns.

### Model Training
- Designs a neural network with layers optimized for binary classification.
- Applies techniques like oversampling or class weighting to handle imbalanced data.

### Evaluation
- **ROC AUC**: Measures the model’s ability to distinguish fraudulent from legitimate transactions.
- **Precision and Recall**: Assesses the balance between false positives and missed fraud cases.
- **Confusion Matrix**: Visualizes true positives, false positives, etc., for detailed analysis.

## Repository Structure
```
credit-card-fraud-detection/
├── fraud_detection.py           # Main script for preprocessing, training, and evaluation
├── requirements.txt             # List of required Python libraries
├── figures/                     # Directory for output visualizations (e.g., ROC curve, confusion matrix)
├── predictions.csv              # Output file with fraud probabilities and binary predictions
├── README.md                    # This file
└── .gitignore                   # Ignores virtual env, data, and model files
```

## Dependencies
The project relies on the following Python libraries (listed in `requirements.txt`):
- numpy
- pandas
- scikit-learn
- tensorflow  # or torch
- matplotlib
- seaborn

Install them using:
```
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

## Contributing
Contributions are welcome! Please submit pull requests or open issues for bugs, feature requests, or improvements.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
