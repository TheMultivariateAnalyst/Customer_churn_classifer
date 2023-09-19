import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

# Load the dataset
data = pd.read_csv("bank-additional-full.csv", sep=";")

# Descriptive statistics
desc_stats = data.describe()

# Missing values check
missing_values = data.isnull().sum()

# Count of each unique value in the column 'y'
y_counts = data['y'].value_counts()

# Determine categorical columns
data_types = data.dtypes
categorical_columns = data_types[data_types == 'object'].index.tolist()

# One-hot encode the categorical columns using pd.get_dummies
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Split the data
X = data_encoded.drop('y_yes', axis=1)
y = data_encoded['y_yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Apply SMOTE with sampling_strategy=0.35
oversampling_percentage = 0.35
smote = SMOTE(sampling_strategy=oversampling_percentage, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Initializing the Logistic Regression model
log_reg = LogisticRegression(max_iter=10000)

# Training the model
log_reg.fit(X_resampled, y_resampled)

# Predicting probabilities on the testing data
y_prob = log_reg.predict_proba(X_test)[:, 1]

# Classifying samples based on the probability cutoff of 0.3
y_pred = (y_prob > 0.3).astype(int)

# Evaluating the model
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(confusion)
print(report)

# Define the hyperparameter grid for logistic regression
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Train a logistic regression model with the best hyperparameters
best_params = {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}

best_log_reg = LogisticRegression(**best_params, max_iter=10000)
best_log_reg.fit(X_resampled, y_resampled)

# Predict with the best model using 0.3 probability cutoff
best_y_prob = best_log_reg.predict_proba(X_test)[:, 1]
best_y_pred_0_3 = (best_y_prob > 0.3).astype(int)

# Evaluate the model
best_report_0_3 = classification_report(y_test, best_y_pred_0_3)

print("Best Hyperparameters:")
print(best_params)
print("\nClassification Report with 0.3 Probability Cutoff:")
print(best_report_0_3)

# Initializing the MLP classifier
mlp_manual = MLPClassifier(
    hidden_layer_sizes=(35, 35, 35),
    activation='relu',
    solver='adam',
    alpha=1,
    max_iter=2000,
    random_state=123
)

# Training the MLP classifier
mlp_manual.fit(X_resampled, y_resampled)

# Predict using the MLP classifier
y_pred_mlp = mlp_manual.predict(X_test)

# Evaluating the MLP classifier
report_mlp = classification_report(y_test, y_pred_mlp)

print("\nMLP Classification Report:")
print(report_mlp)

# Using custom probability cutoff of 0.3 for MLP
y_prob_mlp = mlp_manual.predict_proba(X_test)[:, 1]
y_pred_mlp_0_3 = (y_prob_mlp > 0.3).astype(int)

# Evaluating the MLP classifier with the custom cutoff
report_mlp_0_3 = classification_report(y_test, y_pred_mlp_0_3)

print("\nMLP Classification Report with 0.3 Probability Cutoff:")
print(report_mlp_0_3)
