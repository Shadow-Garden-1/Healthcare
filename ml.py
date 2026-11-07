import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



data = pd.read_csv("healthcare_dataset.csv")

print(data.head())
print(data.describe())



data.hist(bins=50, figsize=(20, 15))
plt.show()



train_set, test_set = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

print("Training records:", len(train_set))
print("Testing records:", len(test_set))


features = [
    "Age",
    "Billing Amount",
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Medication"
]

X_train = train_set[features]
X_test = test_set[features]

y_train = train_set["Test Results"]
y_test = test_set["Test Results"]



num_features = ["Age", "Billing Amount"]

cat_features = [
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Medication"
]



num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])



cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])



preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

print("Data preprocessing completed!")



model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])


model.fit(X_train, y_train)

print("Model training completed!")



predictions = model.predict(X_test)

print("First 10 predictions:")
print(predictions[:10])



accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy * 100, "%")

print(classification_report(y_test, predictions))


joblib.dump(model, "healthcare_model.pkl")

print("Model saved successfully!")