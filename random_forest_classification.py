import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the data
healthy_df = pd.read_csv('/Users/jackxia/Desktop/Python/healthy.csv')
tumor_df = pd.read_csv('/Users/jackxia/Desktop/Python/tumor.csv')

# Add a label column
healthy_df['label'] = 0  # Label for healthy tissue
tumor_df['label'] = 1    # Label for tumor tissue

# Combine the data
data = pd.concat([healthy_df, tumor_df], axis=0)

# Separate features and labels
X = data.drop(columns=['label', 'Filename'])  # Drop Filename if present
y = data['label']

# Define the k-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=1)

# Define the model
model = RandomForestClassifier()

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=kf)

# Print the performance
print(f'Cross-Validation Scores: {scores}')
print(f'Mean Cross-Validation Score: {scores.mean()}')

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Define the GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1)

# Perform the grid search
grid_search.fit(X, y)

# Print the best parameters and the best score
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Cross-Validation Score: {grid_search.best_score_}')

# Train the final model with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X, y)

# Save the final model
joblib.dump(best_model, 'random_forest_model.pkl')
