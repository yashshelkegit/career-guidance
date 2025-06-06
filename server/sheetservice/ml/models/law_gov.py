import pandas as pd
df=pd.read_csv(r"C:\Users\ACER\OneDrive\Desktop\Models\data\law_government_scores.csv")

df=df.drop(columns="Email")

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

# Define the relevant features and target column
X = df[['Social Studies', 'Verbal Ability', 'Reasoning Skills', 'Debate', 
        'Public Speaking', 'Research/Ideation', 'Communication', 'Led any team', 'Mental strength']]

# Define the target column
y = df['Law_Government_Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline: scaling numerical columns (StandardScaler) and one-hot encoding for categorical columns (if any)
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = ['Led any team', 'Mental strength']  # Categorical columns

# Create a column transformer that scales numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Scale numerical columns
        ('cat', OneHotEncoder(), categorical_cols)   # One-hot encode the categorical columns
    ]
)

# Define the regressors to compare
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
    
    # Fit the model on the training data
    pipeline.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Print the results for each model
    print(f'{name}:')
    print(f'  R² Score: {r2:.4f}')
    print(f'  Mean Absolute Error: {mae:.4f}')
    print('-' * 40)



pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),  # Preprocessing step (StandardScaler, OneHotEncoder)
    ('model', SVR())  # SVR model
])
pipeline.fit(X_train, y_train)

import pickle

# Save the pipeline to a pickle file
with open('law.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

def predict(input_dict):
    # Load the trained model from pickle
    with open('law.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    relevant_features = [
    'Social Studies',
    'Verbal Ability',
    'Reasoning Skills',
    'Debate',
    'Public Speaking',
    'Research/Ideation',
    'Communication',
    'Led any team',
    'Mental strength'
    ]
    # Prepare the input data
    data = {key: input_dict[key] for key in relevant_features}
    df = pd.DataFrame([data])
    
    # Make prediction
    prediction = pipeline.predict(df)[0]
    return prediction