import pandas as pd
df=pd.read_csv(r"C:\Users\ACER\OneDrive\Desktop\Models\data\science_healthcare_scores.csv")

df=df.drop(columns="Email")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

X = df.drop('Science_Healthcare_Score', axis=1)

# Define the target column
y = df['Science_Healthcare_Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline: scaling numerical columns (StandardScaler) and one-hot encoding for categorical columns (if any)
# Identify numerical columns (you may need to adjust depending on your data)
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = ['Discipline']
# Create a column transformer that scales numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Scale numerical columns
        ('cat', OneHotEncoder(), categorical_cols)   # One-hot encode the 'Discipline' column
    ]
)

# Create a pipeline that combines preprocessing and RandomForestRegressor
regressors = [
    ('Linear Regression', LinearRegression()),
    ('Decision Tree', DecisionTreeRegressor(random_state=67)),
    ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=67)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=67)),
    ('Support Vector Regressor', SVR()),
    ('K-Nearest Neighbors', KNeighborsRegressor(n_neighbors=5))
]

# Iterate through each regressor
for name, model in regressors:
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f'{name}:')
    print(f'  R² Score: {r2:.4f}')
    print(f'  Mean Absolute Error: {mae:.4f}')
    print('-' * 40)




pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),  # Preprocessing step (StandardScaler, OneHotEncoder)
    ('model', KNeighborsRegressor(n_neighbors=5))  # KNN model with 5 neighbors
])
pipeline.fit(X_train, y_train)

import joblib

# Save the trained pipeline to a file
joblib.dump(pipeline, 'healthcare_knn_model.pkl')

def predict(input_dict):
    # Load the trained model from joblib
    pipeline = joblib.load('healthcare_knn_model.pkl')
    relevant_features = [
    'Biology',
    'Chemistry',
    'Mathematics',
    'Research/Ideation',
    'Logical Thinking',
    'Reasoning Skills',
    'Discipline'
    ]

    # Prepare the input data
    data = {key: input_dict[key] for key in relevant_features}
    df = pd.DataFrame([data])
    
    # Make prediction
    prediction = pipeline.predict(df)[0]
    return prediction

predict()