import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_csv("aptitude_dataset.csv")  

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=4, random_state=42))
])

# Train the pipeline
pipeline.fit(df)

competency_labels_map = {
    0: "Specialized",
    1: "Advanced",
    2: "Intermediate",
    3: "Beginner"
}

combined_model = {
    "pipeline": pipeline,
    "labels_map": competency_labels_map
}

with open("clustering_model.pkl", "wb") as file:
    pickle.dump(combined_model, file)

print("Model and labels saved to 'clustering_model.pkl'")


