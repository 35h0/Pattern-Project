import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load Dataset from Local Files
# Place 'student-mat.csv' and 'student-por.csv' in the same directory as this script
mat_path = "D:\pattern_ass\student_data\student-mat.csv"
por_path = "D:\pattern_ass\student_data\student-por.csv"

# Read with delimiter=';'
math_df = pd.read_csv(mat_path, sep=';')
por_df  = pd.read_csv(por_path, sep=';')
# Use math dataset for this project
data = math_df.copy()

# 2. Exploratory Data Analysis (EDA)
print("Dataset shape:", data.shape)
print(data.head())
print(data.info())
print(data.describe())

plt.figure(figsize=(6,4))
plt.hist(data['G3'], bins=range(0,21), edgecolor='k')
plt.title('Distribution of Final Grades (G3)')
plt.xlabel('Final Grade')
plt.ylabel('Count')
plt.show()

# 3. Preprocessing
# Create binary target: pass if G3 >= 10
data['pass'] = np.where(data['G3'] >= 10, 1, 0)

# Drop original grade columns to avoid leakage
target = 'pass'
features = data.drop(columns=['G1','G2','G3','pass']).columns

# Encode categorical features
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# Scale numeric features
numeric_cols = data.select_dtypes(include=['int64','float64']).columns.drop(target)
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# 4. Prepare train/test sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Model Training & Evaluation
def evaluate_model(name, model):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds), "\n")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Gaussian Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}
for name, mdl in models.items():
    evaluate_model(name, mdl)

# 6. Decision Tree Visualization
plt.figure(figsize=(20,10))
plot_tree(models['Decision Tree'], feature_names=features, class_names=['Fail','Pass'], filled=True)
plt.title('Decision Tree Structure')
plt.show()

# 7. Hyperparameter Tuning (Decision Tree)
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

# 8. Conclusions
print("All models evaluated. Logistic Regression and Naive Bayes are fast and interpretable, Decision Tree shows decision rules and visual insights.")
