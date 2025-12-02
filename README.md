# 📡 Telecom Customer Churn Prediction — ML + Streamlit App

An end-to-end machine learning project to **predict telecom customer churn**, **analyze churn patterns**, and **visualize insights** using Python, Scikit-Learn, and Streamlit.

---

## 📌 Table of Contents
- [Overview](#-overview)  
- [Business Problem](#-business-problem)  
- [Dataset](#-dataset)  
- [Tools & Technologies](#-tools--technologies)  
- [Project Structure](#-project-structure)  
- [Data Preparation](#-data-preparation)  
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)  
- [Machine Learning Model](#-machine-learning-model)  
- [Streamlit Application](#-streamlit-application)  
- [How to Run This Project](#-how-to-run-this-project)  
- [Final Recommendations](#-final-recommendations)  
- [Author](#-author)  

---

## 🚀 Overview
This project builds a **Telecom Customer Churn Prediction System** that allows users to:  
- Predict whether a customer will churn  
- Analyze customer behavior and trends  
- Explore an interactive churn dashboard  
- View segment-wise churn insights  
- Understand churn drivers via EDA  

The Streamlit app includes **login**, **prediction page**, and **dashboard visualizations** for real-world telecom features.

---

## 🧩 Business Problem
Customer churn causes **revenue loss** in telecom companies. Predicting and preventing churn is key for:  
- Increasing customer retention  
- Reducing revenue loss  
- Targeting at-risk customers  
- Designing personalized retention strategies  

Project objectives:  
- Identify patterns causing churn  
- Predict churn probability for new/existing customers  
- Analyze billing, usage, and service-related churn indicators  
- Build dashboards for actionable insights  

---

## 📊 Dataset
**File:** `Customer Churn.csv`  
Features include:

**Demographics:** Gender, SeniorCitizen, Partner, Dependents  
**Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, TechSupport, DeviceProtection, StreamingTV, StreamingMovies  
**Account Info:** Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Tenure  
**Target:** Churn (Yes/No)  

---

## 🛠 Tools & Technologies

| Category | Tools Used |
|----------|------------|
| Programming | Python |
| ML / Data Pipeline | Scikit-Learn, Pandas, NumPy |
| Dashboarding | Streamlit, Plotly |
| Model Saving | Pickle |
| Notebooks | Jupyter Notebook |
| Visualization | Plotly Express, Graph Objects |
| Version Control | GitHub |

---

## 📁 Project Structure

```
telecom_churn_prediction/
│
├── app.py                  # Streamlit application
├── customer_churn_model.pkl # Trained ML model
├── encoders.pkl            # Label encoders for categorical features
├── Customer Churn.csv      # Dataset
│
├── notebooks/
│   └── Eda_analysis.ipynb  # Full exploratory data analysis
│
├── README.md
└── requirements.txt
```

---

## 🧹 Data Preparation
- Handle missing `TotalCharges` values  
- Convert `TotalCharges` to numeric  
- Encode categorical features with `LabelEncoder`  
- Handle “No internet service” categories  
- Standardize text inputs  
- Split dataset into train/test  
- Save trained model & encoders as Pickle files  

---

## 🔍 Exploratory Data Analysis (EDA)
**Key Insights:**  
- Churn rate: ~26.5%  
- Month-to-Month contracts have highest churn  
- High churn among Fiber Optic users  
- Add-on services (Tech Support, Online Security) reduce churn  
- Tenure < 6 months = highest churn segment  
- High MonthlyCharges increases churn probability  

**Visual Highlights:**  
- Bar chart: Churn concentrated among Month-to-Month customers  
- Histogram: New customers (0–3 months) form biggest churn group  
- Boxplot: Churners have higher MonthlyCharges  
- Heatmap: Positive correlation between Tenure and TotalCharges  

---

## 🤖 Machine Learning Model
- **Algorithm:** RandomForestClassifier  
- **Performance:**  
  - Accuracy: 82.4%  
  - Recall (Churn): 0.71  
  - Precision: 0.69  
  - F1 Score: 0.70  
- **Important Features:** Contract, Tenure, MonthlyCharges, OnlineSecurity, TechSupport, PaymentMethod  

**Model & encoders stored as Pickle files:**  
`customer_churn_model.pkl`, `encoders.pkl`

---

## 🌐 Streamlit Application

### 1. Login System
- Username: `sravalika`  
- Password: `12345678`  
- Prevents unauthorized access  

### 2. Churn Prediction Page
Input features: Gender, Contract, Billing, Internet Service, Security Services, MonthlyCharges, Tenure, TotalCharges  
Outputs: Churn Prediction (Yes/No) and Churn Probability (%)  

### 3. Interactive Dashboard
- Churn Rate by Tenure Group  
- Tenure Distribution  
- Service Adoption vs Churn  
- InternetService + Contract Sunburst Chart  
- Monthly Charges vs Churn Boxplot  
- Tenure vs MonthlyCharges Scatter Plot  
- Auto-generated KPIs  

### 4. CSV Upload Support
Users can upload **their own dataset** for analysis.

---

## ▶️ How to Run This Project
```bash
git clone https://github.com/sravalika-gudimalla/telecom_churn_prediction.git
cd telecom_churn_prediction
pip install -r requirements.txt
streamlit run app.py
```
**Note:** Keep `customer_churn_model.pkl` and `encoders.pkl` in the project folder.

---

## 🧠 Final Recommendations
- Improve onboarding for first 3 months  
- Offer discounts for high-billing at-risk customers  
- Promote 1-year & 2-year contracts  
- Improve Fiber Optic service quality  
- Increase adoption of OnlineSecurity & TechSupport services  
- Simplify billing for electronic check users  
- Monitor customer KPIs regularly  

---

## ✨ Author
**Sravalika Gudimella**  
Machine Learning / Data Science Enthusiast  

📧 Email: [sravalika1969@gmail.com](mailto:sravalika1969@gmail.com)  
🔗 GitHub: [https://github.com/sravalika-gudimalla](https://github.com/sravalika-gudimalla)

