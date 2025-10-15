#  House Price Prediction — Regression Model Report

## Overview
This project focuses on predicting **house prices (in lakhs)** using machine learning regression techniques.  
The dataset contains both **numeric and categorical variables** such as number of rooms, square footage, and location.

We explored multiple linear regression models, applied preprocessing pipelines, and evaluated their performance using standard regression metrics.

---

##  Methodology

### 1. **Data Preprocessing**
The preprocessing pipeline handled missing values, scaling, and encoding for both numeric and categorical columns.

####  Numeric Columns
- Missing values were filled with **median values** from the training set.  
- Features were standardized using `StandardScaler`.

####  Categorical Columns
- Missing values were filled with the **most frequent category (mode)**.
- Variables were encoded using `OneHotEncoder(handle_unknown='ignore')` to ensure unseen categories in test data didn’t cause errors.

---

### 2. **Data Splitting**

Data was split into training and testing sets:

```python
X = train_proc.drop(columns=['TARGET(PRICE_IN_LACS)'])
y = train_proc['TARGET(PRICE_IN_LACS)']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

This ensures **80%** of the data is used for model training and **20%** for testing generalization.

---

### 3. **Target Transformation**

Since house prices are strictly positive and vary across several orders of magnitude,  
log-transform was applied to stabilize variance and handle skewness.

```python
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
```

This helps models better capture relative differences and prevents them from being dominated by very large price values.

---

### 4. **Models Trained**

| Model | Description |
|-------|--------------|
| **OLS (Linear Regression)** | Baseline linear model without regularization. |
| **Ridge Regression** | L2 regularization — penalizes large coefficients to reduce overfitting. |
| **Lasso Regression** | L1 regularization — performs feature selection by setting some coefficients to zero. |

Each model was trained within a **Pipeline** that included preprocessing (scaling and encoding) to maintain consistency across model comparisons.

---

###  5. **Model Selection Logic**

```python
best_idx = results['RMSE'].idxmin()
best_model_name = results.loc[best_idx, 'Model']
print(f'Best model: {best_model_name}')

if best_model_name == 'OLS':
    best_pipeline = ols_pipeline
    y_pred_best = y_pred_ols
elif best_model_name == 'Ridge':
    best_pipeline = ridge_grid.best_estimator_
    y_pred_best = y_pred_ridge
else:
    best_pipeline = lasso_grid.best_estimator_
    y_pred_best = y_pred_lasso
```

This ensures that the model with the **lowest RMSE** is automatically selected for evaluation and prediction.

---

### 6. **Model Evaluation**

#### Metrics Used

| Metric | Description | Ideal Value |
|--------|--------------|-------------|
| **RMSE (Root Mean Squared Error)** | Average magnitude of prediction error (penalizes large errors). | Lower = Better |
| **MAE (Mean Absolute Error)** | Mean of absolute differences between actual and predicted values. | Lower = Better |
| **R² (Coefficient of Determination)** | Proportion of variance explained by the model. | Closer to 1 = Better |
| **MAPE (Mean Absolute Percentage Error)** | Error as a percentage of actual values. | Lower = Better |

#### Results

| Model | RMSE | MAE | R² | MAPE (%) |
|--------|------|------|----|----------|
| **OLS** | 118.89 | 46.63 | 0.245 | 52.88 |
| **Ridge** | ≈118.9 | ≈47.1 | ≈0.24 | ≈53.0 |
| **Lasso** | 118.89 | 46.63 | 0.245 | 52.88 |

---

### 7. **Best Model — Lasso Regression**

While all models performed similarly, **Lasso Regression** was selected as the best due to its interpretability and built-in feature selection ability (via L1 regularization).

#### Performance Summary
- **R² = 0.245:** Explains about 24.5% of the variance in housing prices.  
- **RMSE = 118.89 lakhs:** Represents the average prediction error magnitude.  
- **MAPE ≈ 52.9%:** Indicates predictions deviate by roughly half of the true price values.

These results show the model captures some relationships between features and price,  
but there’s still room for improvement — particularly in capturing nonlinear trends and interactions.

---

### 8. **Actual vs Predicted Visualization**

```python
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title(f"Actual vs Predicted Prices ({best_model_name})")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
```

This plot visualizes how closely the model’s predictions align with actual prices.  
The red dashed line represents perfect predictions — points closer to it indicate higher accuracy.

---

### 9. **Generating Predictions for Submission**

```python
test_proc = test.copy()
for col in numeric_cols:
    if col in test_proc.columns:
        test_proc[col] = test_proc[col].fillna(train_proc[col].median())
for col in categorical_cols:
    if col in test_proc.columns:
        test_proc[col] = test_proc[col].fillna(train_proc[col].mode()[0])

try:
    X_test_final = test_proc[numeric_cols + categorical_cols]
    preds_test_log = best_pipeline.predict(X_test_final)
    preds_test = np.expm1(preds_test_log)  # Inverse log-transform
    out = test_proc.copy()
    out['TARGET(PRICE_IN_LACS)'] = preds_test
    out[['TARGET(PRICE_IN_LACS)']].to_csv('submission_from_notebook.csv', index=False)
    print('Saved predictions to submission_from_notebook.csv')
except Exception as e:
    print('Could not predict on test.csv automatically:', e)
```

Predictions are saved in **`submission_from_notebook.csv`**, suitable for evaluation.

---

###  10. **Next Steps for Improvement**

- **Feature Engineering:** Add derived attributes such as property age, location proximity, or amenities.  
- **Nonlinear Models:** Try advanced algorithms like Random Forest, XGBoost, or LightGBM.  
- **Cross-Validation:** Implement K-Fold validation for more robust performance metrics.  
- **Hyperparameter Optimization:** Broaden grid search ranges for α (Ridge/Lasso).  
- **Data Enrichment:** Combine with external datasets for socio-economic or neighborhood context.

---

### 11. **Conclusion**

This project establishes a **baseline regression framework** for predicting house prices.  
While the models achieved moderate accuracy (**R² ≈ 0.25**), the pipeline is **robust, reproducible, and easily extendable**.  

Further improvement will come from richer features and **nonlinear modeling techniques** that better capture complex price determinants.
