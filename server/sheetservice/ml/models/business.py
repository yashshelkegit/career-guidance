import pandas as pd

df = pd.read_csv(r"c:\Users\ACER\OneDrive\Desktop\Models\data\business_management_scores.csv")

print(df.head())

df=df.drop(columns="Email")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Assuming df is the DataFrame containing the data
X = df.drop('Business_Management_Score', axis=1)
y = df['Business_Management_Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline: scaling numerical columns (StandardScaler) and one-hot encoding for categorical columns (if any)
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = ['Led any team','Player','Personality','Mental strength','Preferred Nation','Emotional']  # Categorical column 'Yes/No'

# Create a column transformer that scales numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Scale numerical columns
        ('cat', OneHotEncoder(), categorical_cols)   # One-hot encode the 'Use design software' column
    ]
)

# Define the regressors
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
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Print the results
    print(f'{name}:')
    print(f'  R² Score: {r2:.4f}')
    print(f'  Mean Absolute Error: {mae:.4f}')
    print('-' * 40)


import pickle
with open("business.pkl", "wb") as file:
    pickle.dump(pipeline, file)


from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),  # Preprocessor: handles scaling, encoding
    ('model', KNeighborsRegressor(n_neighbors=5))  # KNN with 5 neighbors
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

import pandas as pd

def predict(input_dict):
    # Load the pipeline from file
    with open("business.pkl", "rb") as file:
        pipeline = pickle.load(file)
    relevant_features = [
    'Economics/Finance',
    'Communication',
    'Public Speaking',
    'Reasoning Skills',
    'Led any team',
    'Player',
    'Personality',
    'Mental strength',
    'Preferred Nation',
    'Emotional'
    ]
    # Extract only the relevant features
    data = {key: input_dict[key] for key in relevant_features}
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Predict
    prediction = pipeline.predict(df)[0]
    return prediction


