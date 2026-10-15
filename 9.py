#Write a program using the K-Nearest Neighbors (KNN) algorithm to classify the road surface as Smooth or Rough based on sensor readings.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# CSV columns:
# speed, accel_rms, vibration_rms, roughness_index, surface
data = pd.read_csv("road_sensor.csv")

features = ["speed", "accel_rms", "vibration_rms", "roughness_index"]
X = data[features]
y = data["surface"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance"))
])

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

new_reading = pd.DataFrame(
    [[60, 0.35, 0.42, 3.1]],
    columns=features
)
print("Predicted road surface:", model.predict(new_reading)[0])
