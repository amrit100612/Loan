**A heartfelt thank you to my amazing partners for their valuable contributions, support, and collaboration throughout this project 🙏**

### 🌟 **@srishtik-ui**
### 🌟 **@amolsingh05**

**Your efforts played a crucial role in making this project successful. 🚀**


# 🏦 Loan Approval Prediction System

A Machine Learning–powered system that predicts whether a loan application will be **Approved** or **Rejected** based on applicant financial and demographic information.

This project demonstrates an end-to-end ML pipeline — from data preprocessing and exploratory analysis to model training, evaluation, and deployment using an interactive web application.

---

## 🚀 Project Objective

To build a robust **classification model** that predicts loan approval status using applicant-level financial and demographic attributes, helping financial institutions automate and improve decision-making.

---

## 📊 Dataset Overview

* **Total Records:** 614
* **Input Features:** 13
* **Target Variable:** `Loan_Status`

### 🔑 Key Features

| Feature           | Description                         |
| ----------------- | ----------------------------------- |
| Gender            | Applicant gender                    |
| Married           | Marital status                      |
| Dependents        | Number of dependents                |
| Education         | Education level                     |
| Self_Employed     | Employment type                     |
| ApplicantIncome   | Primary applicant income            |
| CoapplicantIncome | Co-applicant income                 |
| LoanAmount        | Loan amount requested               |
| Loan_Amount_Term  | Loan repayment duration             |
| Credit_History    | Credit history record (0/1)         |
| Property_Area     | Urban/Semiurban/Rural               |
| Loan_Status       | Target variable (Approved/Rejected) |

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Imbalanced Data Handling:** SMOTE
* **Deployment:** Streamlit

---

## 🔄 Machine Learning Workflow

### 1️⃣ Data Preprocessing

* Handling missing values
* Encoding categorical variables (Label Encoding / One-Hot Encoding)
* Feature scaling (if required)
* Outlier detection & treatment

---

### 2️⃣ Exploratory Data Analysis (EDA)

* Distribution plots
* Correlation heatmap
* Loan approval trends
* Credit history impact analysis
* Class imbalance detection
* SMOTE for balancing dataset

---

### 3️⃣ Model Building

The following classification models were implemented:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (SVM)

---

### 4️⃣ Model Evaluation

Models were evaluated using:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report
* Cross-validation

---

## 📈 Best Performing Model

🏆 **Random Forest Classifier**

After hyperparameter tuning, Random Forest achieved the highest accuracy and balanced precision-recall performance, making it the final selected model.

---

## 📂 Project Structure

```
Loan-Approval-Prediction/
│
├── data/
│   └── loan_data.csv
│
├── notebooks/
│   └── EDA_and_Model_Training.ipynb
│
├── models/
│   └── random_forest_model.pkl
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

---

## 💻 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/loan-approval-prediction.git
cd loan-approval-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 🌐 Deployment

The model is deployed using **Streamlit**, allowing users to input applicant details and receive real-time loan approval predictions.

---

## 📌 Future Improvements

* Hyperparameter optimization using GridSearchCV
* Feature selection techniques
* Model explainability (SHAP / LIME)
* Integration with database systems
* Deployment on cloud platforms (AWS / Heroku / GCP)

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repository and submit a pull request.

---


### ⭐ If you found this project helpful, consider giving it a star!
