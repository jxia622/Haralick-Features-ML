import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
import joblib

# Load the data
healthy_df = pd.read_csv('/Users/jackxia/Desktop/Python/healthy.csv')
tumor_df = pd.read_csv('/Users/jackxia/Desktop/Python/tumor.csv')

# Add a label column
healthy_df['label'] = 0
tumor_df['label'] = 1

# Combine the data
data = pd.concat([healthy_df, tumor_df], axis=0)

# Separate features and labels
X = data.drop(columns=['Filename', 'label'])
y = data['label']

# Define the k-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=1)

# Define the model
svm_model = SVC()

# Perform cross-validation
svm_scores = cross_val_score(svm_model, X, y, cv=kf)

# Print the performance
print(f'SVM Cross-Validation Scores: {svm_scores}')
print(f'SVM Mean Cross-Validation Score: {svm_scores.mean()}')

# Define the parameter grid for hyperparameter tuning
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf']
}

# Define the GridSearchCV
svm_grid_search = GridSearchCV(estimator=svm_model, param_grid=svm_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)

# Perform the grid search
svm_grid_search.fit(X, y)

# Print the best parameters and the best score
print(f'SVM Best Parameters: {svm_grid_search.best_params_}')
print(f'SVM Best Cross-Validation Score: {svm_grid_search.best_score_}')

# Train the final model with the best parameters
best_svm_model = svm_grid_search.best_estimator_
best_svm_model.fit(X, y)

# Save the final model
joblib.dump(best_svm_model, 'svm_model.pkl')
