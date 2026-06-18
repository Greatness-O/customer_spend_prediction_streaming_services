## Predicting Customer Spending Behaviour in Streaming Services 

## Overview

This project investigates how machine learning can be used to understand and predict customer behaviour in a streaming platform context. It covers three complementary tasks:

1. **Regression**: Predicting how much a customer spends per month
2. **Classification**: Identifying which customers are likely to churn
3. **Clustering**: Discovering natural customer segments

The dataset contains demographic and transactional records for 5,000 streaming service customers, including satisfaction scores, subscription length, discount usage, support ticket history and regional information.

## Tools Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- tensorflow
- Jupyter Notebook

## Key Findings
**Regression insights:**
- Satisfaction Score is by the most informative single predictor (R² = 0.53).
- Adding all numerical variables lifts R² to 0.86; appending the Region variable further improves it to ~0.87.
- The ANN (R² ≈ 0.875) performs well but does not surpass the best linear model on this dataset.

**Classification insights:**
- The Random Forest classifier correctly identifies 98% of customers, with an AUC-ROC of ~0.99.
- Logistic Regression (AUC ~0.81) is the weakest classifier, suggesting non-linear decision boundaries are important.

**Clustering insights:**
- Three customer segments emerge consistently across methods.
- Subscription Length is a more discriminating clustering feature than Satisfaction Score or Discount Offered.


## Repository Structure
```
customer_spend_prediction_streaming_services/
│
├── README.md
│
├── data/
│   └── Streaming.csv
│     
├── notebook/
│   └── customer_spend_prediction_streaming_services.ipnyb
│
└── outputs/
    └── figures/
      ├── Confusion Matrix_K-Nearest Neighbours.png
      ├──Confusion Matrix_Logistic Regression.png
      ├──Confusion Matrix_Random Forest.png
      ├──agglomerative_Discount_Offered_+_Monthly_Spend.png
      ├──agglomerative_Satisfaction_Score_+_Monthly_Spend.png
      ├──agglomerative_Subscription_Length_+_Monthly_Spend.png
      ├──ann_actual_predicted.png
      ├──churn_label_distribution.png
      ├──correlation_matrix.png
      ├──feature_importances.png
      ├──kMeans_clustering.png
      ├──multivariate_linear.png
      ├──optimal_k.png
      ├──spend_categorical_box.png
      ├──spend_histogram.png
      ├──spend_satisfaction.png
      └──training_curves.png
