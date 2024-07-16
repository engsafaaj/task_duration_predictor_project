import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump

# Load the new data
new_data_path = 'data.csv'
new_data = pd.read_csv(new_data_path)

# Display the first few rows of the new data (if needed for debugging)
# print(new_data.head())

# Preprocess the data
X = new_data.drop(columns='task_duration')
y = new_data['task_duration']

categorical_features = ['project_type', 'task_name']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

numerical_features = ['project_size', 'project_duration']
numerical_transformer = StandardScaler()

text_features = 'task_description'
text_transformer = TfidfVectorizer()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('txt', text_transformer, 'task_description')
    ])

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

# Save the trained model
dump(model, 'task_duration_predictor_new.joblib')

# Output model performance metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
