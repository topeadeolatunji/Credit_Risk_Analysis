# Credit_Risk_Analysis

## Overview
The purpose of this module analysis is to utilize Machine Learning statistical algorithms to make predictions based on data patterns provided. I focused on supervised learning using a free dataset from LendingClub, a peer-to-peer lending services company to evaluate and predict the credit risk of its loans.

To complete this analysis, I used different Machine Learning techniques to train and evaluate the data with unbalanced classes. The dataset from the LendingClub has an unbalanced classification problem due to the number of good loans outweighing the amount of risky loans. In order balance out the classifications to allow for more meaningful predictions and improve the accuracy score, I employed various Machine Learning algorithms to resample the data.

## Results
I used Machine Learning to resample the dataset using Python libraries: scikit-learn and imbalanced-learn evaluate the results and provide a comparison for our analysis.

The original dataset contained 115,675 loan applications in Q1 of 2019. We used the "loan status" to determine whether the application was considered "low" or "high" risk. Applications that had "current" as the "loan status" were classified as "low risk" and the remaining as "high risk". This reduced the dataset to 68,817 total applications with 99% classified as "low risk".

<img width="544" alt="image" src="https://user-images.githubusercontent.com/104689265/189567426-24357b0d-c9f5-4d3a-8300-87aae21e4004.png">

Using the 75/25% method to split the data for training vs. testing, 51,366 "low risk" and 246 "high risk" applications were categorized into the training set.

<img width="953" alt="image" src="https://user-images.githubusercontent.com/104689265/189567672-61df20ee-e29e-476b-95d8-bcb918ccef35.png">

### Deliverable 1: Use Resampling Models to Predict Credit Risk

Oversampling

RandomOverSampler Model randomly selects from the minority class and adds it to the training set until both classifications are equal. The results classified 51,366 records each as High Risk and Low Risk.

<img width="953" alt="image" src="https://user-images.githubusercontent.com/104689265/189567916-579d0a30-f539-4d57-b9a7-c022eb4372f2.png">

Balanced accuracy score: 65%.
<img width="918" alt="image" src="https://user-images.githubusercontent.com/104689265/189568066-682ec057-07bb-4d94-8799-ed019d5dab4e.png">

The "High Risk" precision rate was only 1% with the recall at 72% giving this model an F1 score of 2%.
"Low Risk" had a precision rate of 100% and recall at 59%.

<img width="953" alt="image" src="https://user-images.githubusercontent.com/104689265/189568410-68ba60e2-ad89-4df4-90f9-4a5f4c6edd4d.png">


SMOTE (Synthetic Minority Oversampling Technique) Model, like RandomOverSampler increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection.

The balanced accuracy score improved slightly to 66.2%.
<img width="752" alt="image" src="https://user-images.githubusercontent.com/104689265/189568706-e5ed70b4-6606-4f9d-9482-b627f0faa17f.png">

Like RandomOverSampler, the "High Risk" precision rate again was only 1% with the recall degraded to 63% giving this model an F1 score of 2%.
"Low Risk" had a precision rate of 99% and an improved recall at 69%.

<img width="953" alt="image" src="https://user-images.githubusercontent.com/104689265/189568805-07b53943-017d-426e-824a-f4be56c970a7.png">


Undersampling

ClusterCentroids Model, an algorithm that identifies clusters of the majority class to generate synthetic data points that are representative of the clusters. The model classified 246 records each as High Risk and Low Risk.

<img width="953" alt="image" src="https://user-images.githubusercontent.com/104689265/189569074-5253354f-69f0-4f02-9a4a-419c55f882ad.png">

Balanced accuracy score was lower than the oversampling models at 54.5%.
<img width="740" alt="image" src="https://user-images.githubusercontent.com/104689265/189569180-e8b0920a-f4af-4225-b859-8bd21b5099f9.png">

The "High Risk" precision rate again was only at 1% with the recall at 69% giving this model an F1 score of 1%.
"Low Risk" had a precision rate of 99% and with a lower recall at 40% compared to the oversampling models.
undercm

<img width="953" alt="image" src="https://user-images.githubusercontent.com/104689265/189569251-0de32184-985a-4eef-85f8-bd3e9e3533b2.png">

### Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk

Combination Sampling

SMOTEENN (Synthetic Minority Oversampling Technique + Edited NearestNeighbors) Model combines aspects of both oversampling and undersampling. The model classified 68,460 records as High Risk and 62,011 as Low Risk.

<img width="840" alt="image" src="https://user-images.githubusercontent.com/104689265/189569430-f49bb89e-b552-48be-bdb0-7837fb240cae.png">


The balanced accuracy score improved to 64.7% when using a combined sampling model.


The "High Risk" precision rate did not improve was only 1%, however the recall increased to 72% giving this model an F1 score of 2%.
"Low Risk" still showed a precision rate of 99% with the recall at 57%.

<img width="953" alt="image" src="https://user-images.githubusercontent.com/104689265/189569580-bf30c778-e142-4183-a806-b15b5ff616fd.png">


### Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk

Compare two new Machine Learning models that reduce bias to predict credit risk. The models classified 51,366 as High Risk and 246 as Low Risk.

<img width="814" alt="image" src="https://user-images.githubusercontent.com/104689265/189569770-32cc3bd5-f112-4774-a404-d721a24954f3.png">

BalancedRandomForestClassifier Model, two trees of the same size and equal size to the minority class are constructed to represent one for the majority class and one for the minority class.

The balanced accuracy score increased to 78.9% for this model.
<img width="953" alt="image" src="https://user-images.githubusercontent.com/104689265/189569835-aee829b9-ed78-42e1-9fad-cf7601b2de01.png">

The "High Risk precision rate increased to 3% with the recall at 70% giving this model an F1 score of 6%.
"Low Risk" still had a precision rate of 100% with the recall at 87%.
The top feature by importance was "total_rec_prncp" at 7.9% of the total.

<img width="953" alt="image" src="https://user-images.githubusercontent.com/104689265/189569931-600e1f2c-deee-490d-be00-b5d088e2e5de.png">

<img width="408" alt="image" src="https://user-images.githubusercontent.com/104689265/189570220-1a82fd2f-7053-4680-b5ad-ae397718950c.png">


The balanced accuracy score increased to 93.2% with this model.
<img width="352" alt="image" src="https://user-images.githubusercontent.com/104689265/189570284-40244dc1-d8d3-40c2-8479-18754e19942a.png">

The "High Risk precision rate increased to 9% with the recall at 92% giving this model an F1 score of 16%.
"Low Risk" still had a precision rate of 100% with the recall now at 94%.

<img width="953" alt="image" src="https://user-images.githubusercontent.com/104689265/189570372-0ee0ea4d-b3dc-4043-b858-31845f5e6ec6.png">

## Summary
In reviewing all six models, the EasyEnsembleClassifer model yielded the best results with an accuracy rate of 93.2% and a 9% precision rate when predicting "High Risk candidates. The sensitivity rate (aka recall) was also the highest at 92% compared to the other models. The result for predicting "Low Risk" was also the highest with the sensitivity rate at 94% and an F1 score of 97%. Therefore, if a model needed to be recommended to perform this type of analysis, then this one would be the clear choice.

### Ranking of models in descending order based on "High Risk" results:

EasyEnsembleClassifer: 93.2% accuracy, 9% precision, 92% recall, and 16% F1 Score
BalancedRandomForestClassifer: 78.9% accuracy, 3% precision, 70% recall and 6% F1 Score
SMOTE: 65.2% accuracy, 1% precision, 61% recall and 2% F1 Score
SMOTEENN: 64.5% accuracy, 1% precision, 72% recall and 2% F1 Score
RandomOverSampler: 64.0% accuracy, 1% precision, 66% recall and 2% F1 Score
ClusterCentroids: 54.5% accuracy, 1% precision, 69% recall and 1% F1 Score
