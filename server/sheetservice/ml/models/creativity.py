import pandas as pd

df = pd.read_csv(r"C:\Users\ACER\OneDrive\Desktop\Models\data\creativity_design_scores.csv")
df=df.drop(columns=["Email","Content pieces per month"])

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
X = df.drop('Creativity_Design_Score', axis=1)
y = df['Creativity_Design_Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline: scaling numerical columns (StandardScaler) and one-hot encoding for categorical columns (if any)
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = ['Use design software']  # Categorical column 'Yes/No'

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


from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

# Define pipeline with SVR model
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),  # Preprocessing step (StandardScaler, OneHotEncoder)
    ('model', SVR())  # SVR model
])
pipeline.fit(X_train, y_train)

# New relevant features


import pickle

# Save the pipeline to a pickle file
with open("creativity.pkl", "wb") as file:
    pickle.dump(pipeline, file)

def predict(input_dict):
    # Load pipeline from pickle file
    with open("creativity.pkl", "rb") as file:
        pipeline = pickle.load(file)
    relevant_features = [
    'Drawing/Art',
    'Creativity',
    'Design Thinking',
    'Music/Dance',
    'Crafting',
    'Writing Skill',
    'Use design software',
    'Mathematics',
    'Logical Thinking'
    ]

    # Extract relevant features
    data = {key: input_dict[key] for key in relevant_features}
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Make prediction
    prediction = pipeline.predict(df)[0]
    return prediction