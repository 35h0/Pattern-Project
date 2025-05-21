import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === 1. Load Dataset ===
mat_path = "D:/pattern_ass/student_data/student-mat.csv"
por_path = "D:/pattern_ass/student_data/student-por.csv"

math_df = pd.read_csv(mat_path, sep=';')
por_df = pd.read_csv(por_path, sep=';')
data = math_df.copy()  # Use math dataset

# === 2. Exploratory Data Analysis ===
print("Dataset shape:", data.shape)
print(data.head())
print(data.info())
print(data.describe())

plt.figure(figsize=(6,4))
plt.hist(data['G3'], bins=range(0, 21), edgecolor='k')
plt.title('Distribution of Final Grades (G3)')
plt.xlabel('Final Grade')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# === 3. Preprocessing ===
# Binary target variable: pass = 1 if G3 >= 10
data['pass'] = np.where(data['G3'] >= 10, 1, 0)

# Drop grade columns to prevent data leakage
target = 'pass'
features = data.drop(columns=['G1', 'G2', 'G3', 'pass']).columns.tolist()

# Encode categorical variables
le = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col])

# Normalize numeric features
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.drop(target)
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# === 4. Train-Test Split ===
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 5. Model Evaluation Function ===
def evaluate_model(name, model):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"=== {name} ===")
    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds), "\n")

# === 6. Train & Evaluate Models ===
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Gaussian Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

for name, mdl in models.items():
    evaluate_model(name, mdl)

# === 7. Decision Tree Visualization ===
plt.figure(figsize=(18, 10))
plot_tree(models['Decision Tree'], feature_names=features, class_names=['Fail', 'Pass'], filled=True)
plt.title('Decision Tree Visualization')
plt.tight_layout()
plt.show()

# === 8. Hyperparameter Tuning (Decision Tree) ===
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)

best_dt = grid.best_estimator_
evaluate_model('Tuned Decision Tree', best_dt)

# === 9. Conclusion ===
print("✅ Evaluation complete. Consider Random Forest or Gradient Boosting if accuracy is a priority.")
