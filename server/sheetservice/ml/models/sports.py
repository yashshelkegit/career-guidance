import pandas as pd
df=pd.read_csv(r"C:\Users\ACER\OneDrive\Desktop\Models\data\sports_defense_scores.csv")

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

# Define the feature columns (exclude 'Sports_Defense_Score')
X = df.drop('Sports_Defense_Score', axis=1)

# Define the target column
y = df['Sports_Defense_Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical columns
numerical_cols = ['Physical fitness']

# Identify categorical columns
categorical_cols = ['Exercise/gym', 'Sports or Outdoor Activities', 'Discipline', 'Mental strength', 'Player', 'Emotional']

# Create a column transformer to scale numerical columns and one-hot encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Standardize numerical columns
        ('cat', OneHotEncoder(), categorical_cols)   # One-hot encode categorical columns
    ]
)

# Create a pipeline with preprocessing and a RandomForestRegressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

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




# Assuming your pipeline is already created and trained with an SVR model
sports_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),  # Assuming preprocessor is already defined
    ('model', SVR())  # SVR model
])

# Train the pipeline if necessary
pipeline.fit(X_train, y_train)  # X_train should have the relevant features

import pandas as pd
import pickle
with open("sports.pkl", "wb") as file:
    pickle.dump(pipeline, file)

def predict(input_dict):
    # Load the saved pipeline from the .pkl file inside the predict function
    with open("sports.pkl", "rb") as file:
        sports_pipeline = pickle.load(file)
    
    # Define the relevant features needed for the prediction
    relevant_features = [
    'Physical fitness',
    'Exercise/gym',
    'Sports or Outdoor Activities',
    'Discipline',
    'Mental strength',
    'Player',
    'Emotional',
    ]
    
    # Extract relevant features from the input dictionary
    data = {key: input_dict[key] for key in relevant_features}
    
    # Convert the data to a DataFrame
    df = pd.DataFrame([data])
    
    # Make prediction using the loaded pipeline
    prediction = sports_pipeline.predict(df)[0]
    return prediction
