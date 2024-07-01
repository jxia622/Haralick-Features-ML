import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
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

# Ensure data is suitable for CNN
X = X.values.reshape(X.shape[0], X.shape[1], 1, 1)

# Define the CNN model
def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Convert labels to categorical
y = to_categorical(y)

# Define the k-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=1)

# Define the KerasClassifier
model = KerasClassifier(build_fn=create_cnn_model, epochs=10, batch_size=32, verbose=0)

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=kf)

# Print the performance
print(f'Cross-Validation Scores: {scores}')
print(f'Mean Cross-Validation Score: {scores.mean()}')

# Train the final model
model.fit(X, y)

# Save the final model
model.model.save('cnn_model.h5')
