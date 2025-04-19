# Telco Customer Churn Prediction

## Overview
This project analyzes **customer churn** in the **telecommunications industry** using **machine learning techniques**. The dataset includes **customer demographics, usage patterns, and account information** to predict churn and help businesses improve customer retention strategies.

## Objective
- Predict whether a customer will **churn or remain** with the company.
- Identify key factors influencing **customer attrition**.
- Improve **marketing and retention strategies** for high-risk customers.

## Dataset Description
The dataset consists of **3333 rows and 21 columns**, capturing various customer attributes.

### Features
- **Account_Length**: Number of days the account has been active.
- **Vmail_Message**: Number of voicemail messages.
- **Day_Mins, Eve_Mins, Night_Mins, Intl_Mins**: Usage in different periods.
- **CustServ_Calls**: Number of customer service calls made.
- **Churn (Target Variable)**: Whether the customer churned (yes/no).
- **Intl_Plan, Vmail_Plan**: Subscription to international/voicemail plans.
- **Day_Calls, Eve_Calls, Night_Calls, Intl_Calls**: Number of calls in each period.
- **Day_Charge, Eve_Charge, Night_Charge, Intl_Charge**: Charges corresponding to usage.
- **State, Area_Code, Phone**: Customer location details.

## Project Workflow

### 1. Data Exploration
- **Checked dataset shape**: `(3333, 21)`, indicating **3333 customers** with **21 attributes**.
- **Checked missing values**: No missing values found.
- **Churn distribution**:  
  - **2850 customers (85%) did not churn**.  
  - **483 customers (15%) churned**.  
- **Visualizations**:
  - **Histogram for Account Length**.
  - **Boxplot for Account Length vs. Churn** → No significant pattern observed.

### 2. Data Preprocessing
- **Converted categorical variables** (`Churn`, `Intl_Plan`, `Vmail_Plan`) to numerical format (`0 = No`, `1 = Yes`).
- **Encoded State variable** using `pd.get_dummies()` to create binary indicators.
- **Dropped Phone column** since it is not useful for modeling.
- **Feature Scaling**: Normalized **continuous variables** for better model performance.

### 3. Correlation Analysis
- **Strong correlations observed**:
  - **Day Mins & Day Charge**: `1.00`
  - **Eve Mins & Eve Charge**: `1.00`
  - **Night Mins & Night Charge**: `1.00`
  - **Intl Mins & Intl Charge**: `0.999`
- **Redundancy elimination**: Removed **one of each highly correlated pair** to avoid multicollinearity.

### 4. Model Building
- **Separated features and target variable**:
  - `features = telco.drop(['Churn'], axis=1)`
  - `target = telco['Churn']`
- **Split dataset** into **training (80%) and testing (20%)**:
  - `train_test_split(features, target, test_size=0.2, random_state=42)`
- **Implemented Logistic Regression**:
  - Used `LogisticRegression()` from `sklearn`.
  - Trained on **X_train, y_train**.
  - Predicted churn using **X_test**.

### 5. Model Evaluation
- **Accuracy Score**: `85.1%`
  - `clf_score = clf.score(X_test, y_test)`
  - Model correctly predicts churn **85% of the time**.

### 6. Key Findings & Business Insights
- **Customer Support Calls & Churn**: Higher customer service calls **correlate with churn**, suggesting poor customer experience.
- **Plan Subscription Impact**: Customers with **International Plan** are **more likely to churn**.
- **No Clear Trend in Tenure**: Account length does not strongly influence churn.

### 7. Next Steps
- **Hyperparameter Tuning**: Optimize Logistic Regression model.
- **Feature Engineering**: Identify more useful predictors.
- **Try Different Models**: Compare with Decision Trees, Random Forest, and XGBoost.

## Author
Rajkumar Shenigaram
