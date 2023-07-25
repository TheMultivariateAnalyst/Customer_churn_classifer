# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# Load the dataset
data = pd.read_csv('bank-additional-full.csv', sep=';')

# Drop the 'duration' column
data = data.drop(['duration'], axis=1)

# Define the feature set X and the target y
X = data.drop(['y'], axis=1)
y = data['y'].apply(lambda x: 0 if x == 'no' else 1)  # Convert the target to binary format (0, 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify the numerical and categorical columns
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(), cat_cols)])

# Fitting and transforming the training data
X_train = preprocessor.fit_transform(X_train)

# Transforming the test data
X_test = preprocessor.transform(X_test)

# Define the Logistic Regression model
lr_model = LogisticRegression(random_state=42, max_iter=1000)

# Train the model
lr_model.fit(X_train, y_train)

# Use cross-validation to evaluate accuracy
cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='accuracy')
mean_cv_score = cv_scores.mean()

# Predict on the test set
y_pred_lr = lr_model.predict(X_test)

# Classification report
report = classification_report(y_test, y_pred_lr, target_names=['No', 'Yes'])

# Print the mean CV score and the classification report
print(mean_cv_score)
print(report)

# Define the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Classification report
report = classification_report(y_test, y_pred_xgb, target_names=['No', 'Yes'])

# Print the classification report
print(report)
