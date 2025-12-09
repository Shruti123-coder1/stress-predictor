import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/stress_dataset.csv")

# Encode target
df["stress_label"] = df["stress_level"].map({"Low":0,"Medium":1,"High":2})

# Split data
X = df[["sleep_hours","heart_rate","work_stress"]]
y = df["stress_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model/stress_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")

# Predict function
def predict_stress(sleep, hr, work):
    with open("model/stress_model.pkl", "rb") as f:
        model = pickle.load(f)
    pred = model.predict([[sleep, hr, work]])[0]
    mapping = {0:"Low 😌", 1:"Medium 😐", 2:"High 😟"}
    return mapping[pred]
