
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# -----------------------------------
# 11. Class distribution
# -----------------------------------
print("\nClass distribution (train)")
print(y_train.value_counts())

print("\nClass distribution (validation)")
print(y_val.value_counts())

print("\nClass distribution (test)")
print(y_test.value_counts())
