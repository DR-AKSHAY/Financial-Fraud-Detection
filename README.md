# Financial Fraud Detection Using Machine Learning

## About the Project

▪ Financial fraud is a growing concern for banks and financial institutions. Every day, a large number of transactions take place, and identifying suspicious activity among them can be challenging.

▪ I developed this project to explore how machine learning can help identify potentially fraudulent credit card transactions.

▪ The application uses machine learning models to analyse transaction data and classify transactions as either    normal or requiring further review. I also created a simple web dashboard where users can upload a CSV file, review transaction results, apply filters and download a report.

⏹ This project was developed as part of my MCA Semester III practical learning.

## Project Objectives

The main objectives of this project are:

 ◽ Understand how machine learning can be used for financial fraud detection.
 ◽ Analyse credit card transaction data.
 ◽ Train and compare different machine learning algorithms.
 ◽ Identify transactions that may require further investigation.
 ◽ Develop a web application to display prediction results.
 ◽ Understand how model performance is evaluated.

## Technologies Used in this project

◼  Technology       Purpose 
▪ Python       -     Main programming language 
▪ Pandas       -     Data handling and analysis 
▪ NumPy        -     Numerical operations 
▪ Matplotlib   -     Data visualisation 
▪ Seaborn      -     Statistical visualisation 
▪ Scikit-learn -     Machine learning and model evaluation 
▪ Streamlit    -     Web application 
▪ Joblib       -     Saving and loading the trained model 

## Dataset

For this project, I used the European Credit Card Fraud Detection dataset available on Kaggle.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains:

- 284,807 credit card transactions.
- 492 fraudulent transactions.
- 284,315 legitimate transactions.
- 30 input features and one target column.

Most of the input features (V1 to V28) are anonymised. The `Class` column identifies the transaction type:

- `0` – Legitimate transaction
- `1` – Fraudulent transaction

The dataset is highly imbalanced because fraudulent transactions represent only a very small percentage of all transactions.

## Machine Learning Models

I experimented with two machine learning algorithms.

### 1. Logistic Regression

Logistic Regression was used as the first classification model. I applied feature scaling using StandardScaler and handled class imbalance using class weights.

### 2. Random Forest

Random Forest was used as the second model. It combines multiple decision trees to make predictions.

Both models were trained and evaluated using the same training and testing split so their results could be compared fairly.

## Model Performance

The dataset was divided into 80% training data and 20% testing data.

The following results were obtained on the held-out test dataset:

| Evaluation Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 97.55% | 99.95% |
| Precision | 6.10% | 90.59% |
| Recall | 91.84% | 78.57% |
| F1-Score | 11.44% | 84.15% |

Random Forest achieved better precision and F1-score, while Logistic Regression achieved higher recall.

Based on these results, I selected Random Forest for the transaction review application. However, the choice of model depends on whether reducing false alarms or detecting as many fraudulent transactions as possible is the higher priority.

## Application Features

The Streamlit application includes the following features:

- Upload transaction data through a CSV file.
- Preview the uploaded transactions.
- Analyse transactions using the trained Random Forest model.
- Display the number of transactions requiring review.
- Show the distribution of transaction predictions.
- Filter transactions by review status.
- View important transaction details.
- Download the prediction results as a CSV report.
- Display model performance and confusion matrix.
- Compare Logistic Regression and Random Forest.

Transactions marked as "Needs Review" are only potential cases requiring further investigation. A model prediction does not confirm that a transaction is fraudulent.

## Project Structure

```text
Financial-Fraud-Detection/
│
├── dataset/
│   └── creditcard.csv
│
├── models/
│   └── fraud_detection_model.pkl
│
├── outputs/
│   └── model_evaluation.json
│
├── notebooks/
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## How to Run the Project

Follow these steps to run the application on your computer.

### Step 1: Clone the repository

```bash
git clone (https://github.com/DR-AKSHAY/Financial-Fraud-Detection.git)
```

### Step 2: Open the project folder

```bash
cd Financial-Fraud-Detection
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install the required libraries

```bash
pip install -r requirements.txt
```

### Step 5: Train the machine learning models

```bash
python train_model.py
```

This will train both models, calculate their evaluation metrics and save the Random Forest model and evaluation results.

### Step 6: Start the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Limitations

There are a few limitations to this project:

- The dataset contains anonymised features, so their original financial meaning is not available.
- The model was trained on a publicly available dataset and may not perform the same way on real-world banking data.
- The dataset is highly imbalanced.
- Some fraudulent transactions may not be identified by the model.
- Transactions marked for review require human verification.

## Conclusion

⩥ Working on this project helped me understand how machine learning can be applied to a real-world problem like financial fraud detection.

⩥ I learned how to explore and prepare a dataset, train machine learning models, compare their performance and build a web application using Streamlit.

⩥ The project also helped me understand why accuracy alone is not enough when working with imbalanced datasets. Precision, recall and F1-score are important when evaluating fraud detection models.

⩥ In the future, this project could be improved by testing additional algorithms, tuning prediction thresholds and evaluating the model on more diverse transaction datasets.

---

**Project:** Financial Fraud Detection Using Machine Learning  
**Application:** Transaction Monitoring  
**Developed using:** Python, Scikit-learn and Streamlit  
**Academic Level:** MCA Semester III
*Developer* Avinash tripathi 
