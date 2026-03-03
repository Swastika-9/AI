import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns

from dropout_preprocessing import load_and_preprocess

# 1. Load preprocessed data
X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess()

# 2. Logistic Regression + GridSearch
lr = LogisticRegression(max_iter=1000, random_state=42)
lr_params = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear']
}
lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='accuracy')
lr_grid.fit(X_train, y_train)
print("\nBest Logistic Regression Parameters:", lr_grid.best_params_)

# 3. Decision Tree + GridSearch
dt = DecisionTreeClassifier(random_state=42)
dt_params = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
dt_grid = GridSearchCV(dt, dt_params, cv=5, scoring='accuracy')
dt_grid.fit(X_train, y_train)
print("Best Decision Tree Parameters:", dt_grid.best_params_)

# 4. Validation Predictions
y_val_pred_lr = lr_grid.predict(X_val)
y_val_pred_dt = dt_grid.predict(X_val)

def print_metrics(y_true, y_pred, model_name):
    print(f"\n{model_name} Metrics:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred))

print_metrics(y_val, y_val_pred_lr, "Logistic Regression")
print_metrics(y_val, y_val_pred_dt, "Decision Tree")

# 5. Confusion Matrices
fig, ax = plt.subplots(1, 2, figsize=(12,5))

cm_lr = confusion_matrix(y_val, y_val_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges', ax=ax[0])
ax[0].set_title("Logistic Regression")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

cm_dt = confusion_matrix(y_val, y_val_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Purples', ax=ax[1])
ax[1].set_title("Decision Tree")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# 6. ROC Curve
plt.figure(figsize=(8,6))
ax = plt.gca()
RocCurveDisplay.from_estimator(lr_grid, X_val, y_val, ax=ax, name="Logistic Regression")
RocCurveDisplay.from_estimator(dt_grid, X_val, y_val, ax=ax, name="Decision Tree")
plt.title("ROC Curve Comparison")
plt.show()

# 7. Decision Tree Visualization
plt.figure(figsize=(18,8))
plot_tree(dt_grid.best_estimator_,
          feature_names=X_train.columns,
          class_names=["Graduate/Enrolled","Dropout"],
          filled=True,
          max_depth=3)
plt.title("Decision Tree (Top Levels)")
plt.show()