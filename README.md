# Customer Churn Prediction

1. Data Background (The Problem)
The Indian telecommunications sector is highly competitive. A major challenge for providers (like Airtel, Jio, Vodafone, BSNL) is **customer churn**—when users stop using their services. 

This project aims to predict customer churn by analyzing two datasets:
* **Demographics**: Customer age, gender, location, dependents, and estimated salary.
* **Usage Patterns**: Number of calls made, SMS sent, and data used.

## 2. Analysis Method (Solution Thinking Flow)
To predict if a customer will churn, we followed this workflow:
1. **Data Integration:** Merged demographic and usage data using `customer_id`.
2. **Exploratory Data Analysis:** Found that the dataset is imbalanced (only ~20% of customers actually churned).
3. **Preprocessing:** * Converted text categories (like gender and city) into numbers using One-Hot Encoding.
   * Scaled numerical features (like salary and data usage) so larger numbers don't unfairly dominate the model.
4. **Modeling:** Split the data into training (80%) and testing (20%) sets. We trained two models: **Logistic Regression** and **Random Forest**.
5. **Evaluation:** Compared models using accuracy, precision, recall, and confusion matrices.

## 3. Libraries Used & Desired Data Structure
Our goal was to transform raw, mixed-type data into a clean, purely numeric dataset that machine learning models can process. We used the following libraries:

* **`pandas`**: To load, merge, and manipulate data (including generating dummy variables for categorical data).
* **`StandardScaler`**: To normalize numerical data so all features are on the same scale.
* **`train_test_split`**: To divide our data for training the model and testing it on unseen data.
* **`LogisticRegression` & `RandomForestClassifier`**: The core machine learning algorithms used to predict churn.
* **`classification_report` & `confusion_matrix`**: To evaluate model performance beyond simple accuracy, showing exactly where the models get things right or wrong.

## 4. Results & Conclusion
* **Logistic Regression:** Reached **73%** accuracy. It was able to identify a small portion of actual churners.
* **Random Forest:** Reached **79%** accuracy, making it the model with the higher overall score.
* **Conclusion:** While **Random Forest** achieved higher raw accuracy, the confusion matrix shows it struggled to identify the minority class (the actual churners) due to the imbalanced data (80% non-churn / 20% churn). 
* **Next Steps:** Future improvements should focus on handling data imbalance (e.g., SMOTE or class weighting) to help the model better catch customers who are about to leave.
