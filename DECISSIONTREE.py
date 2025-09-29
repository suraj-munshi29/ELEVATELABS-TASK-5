import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('heart.csv')

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
print("Decision Tree Classifier trained.")

train_accuracies = []
test_accuracies = []
depths = range(1, 15)

for depth in depths:
    dt_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_model.fit(X_train, y_train)
    train_accuracies.append(accuracy_score(y_train, dt_model.predict(X_train)))
    test_accuracies.append(accuracy_score(y_test, dt_model.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(depths, train_accuracies, 'o-', label='Train Accuracy')
plt.plot(depths, test_accuracies, 'o-', label='Test Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy vs. Max Depth')
plt.legend()
plt.savefig('accuracy_vs_depth.png')
print("Accuracy vs. Depth plot saved as accuracy_vs_depth.png")

rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
rf_classifier.fit(X_train, y_train)

dt_accuracy = accuracy_score(y_test, dt_classifier.predict(X_test))
rf_accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))

print(f"\nDecision Tree Test Accuracy: {dt_accuracy:.4f}")
print(f"Random Forest Test Accuracy: {rf_accuracy:.4f}")

importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.tight_layout()
plt.savefig('feature_importances.png')
print("Feature importances plot saved as feature_importances.png")

dt_cv_scores = cross_val_score(dt_classifier, X, y, cv=5)
rf_cv_scores = cross_val_score(rf_classifier, X, y, cv=5)

print(f"\nDecision Tree Cross-Validation Mean Accuracy: {dt_cv_scores.mean():.4f}")
print(f"Random Forest Cross-Validation Mean Accuracy: {rf_cv_scores.mean():.4f}")
