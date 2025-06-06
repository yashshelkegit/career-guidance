import pandas as pd
df=pd.read_csv(r"C:\Users\ACER\OneDrive\Desktop\Models\data\media_communication_scores.csv")

df=df.drop(columns=["Email","Content pieces per month","Social media screen time"])

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

# Target and features
X = df.drop('Media_Communication_Score', axis=1)
y = df['Media_Communication_Score']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = ['Personality']  # One categorical column: 'Introvert'/'Extrovert'

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)  # Drop first to avoid dummy variable trap
    ]
)

# Define regressors to try
regressors = [
    ('Linear Regression', LinearRegression()),
    ('Decision Tree', DecisionTreeRegressor(random_state=67)),
    ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=67)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=67)),
    ('Support Vector Regressor', SVR()),
    ('K-Nearest Neighbors', KNeighborsRegressor(n_neighbors=5))
]

# Loop through regressors and evaluate
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

import pickle
with open('media_communication_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)



pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),  # Assuming preprocessor is already defined
    ('model', SVR())  # SVR model
])

pipeline.fit(X_train, y_train)  # Ensure X_train contains only the relevant features

import pandas as pd

# Define the predict function
def predict(input_dict):
    # Load the trained pipeline
    with open('media_communication_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    relevant_features = [
    'Language',
    'Communication',
    'Public Speaking',
    'Writing Skill',
    'Acting/Drama',
    'Creativity',
    'Personality'
    ]
    # Get expected feature names from the pipeline (if available)
    try:
        relevant_features = pipeline.feature_names_in_
    except AttributeError:
        raise ValueError("Pipeline does not expose expected input features. Please specify them manually.")

    # Extract relevant features from input_dict
    data = {key: input_dict.get(key, 0) for key in relevant_features}

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Predict
    prediction = pipeline.predict(df)[0]
    return prediction




