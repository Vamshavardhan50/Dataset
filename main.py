import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(df.head())

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
df = df[features + ["Survived"]]

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

pclass = int(input("Enter Passenger Class (1, 2, 3): "))
sex = int(input("Enter Sex (0 = Male, 1 = Female): "))
age = float(input("Enter Age: "))
sibsp = int(input("Enter Number of Siblings/Spouses Aboard: "))
parch = int(input("Enter Number of Parents/Children Aboard: "))
fare = float(input("Enter Fare: "))
embarked = int(input("Enter Embarkation Port (0 = C, 1 = Q, 2 = S): "))

sample_passenger = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
sample_passenger = scaler.transform(sample_passenger)
prediction = model.predict(sample_passenger)
print(f"Survival Prediction (1 = Survived, 0 = Not Survived): {prediction[0]}")
