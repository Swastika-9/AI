import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess():
    # 1. Load Dataset
    df = pd.read_csv("student_dropout.csv") 
    print("Loaded dataset:")
    print(df.head())  

    # 2. Select Features
    selected_cols = [
        "Gender",
        "Age at enrollment",
        "Admission grade",
        "Previous qualification (grade)",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
        "Tuition fees up to date",
        "Debtor",
        "Scholarship holder",
        "Unemployment rate",
        "GDP",
        "Inflation rate",
        "target"
    ]
    df = df[selected_cols]
    df = df.dropna()
    print("\nDataset after feature selection and dropping missing values:")
    print(df.head())

    # 3. Encode Target
    df["target"] = df["target"].map({
        "Dropout": 1,
        "Graduate": 0,
        "Enrolled": 0
    })
    print("\nDataset after target encoding:")
    print(df.head())

    # 4. Split Features and Target
    X = df.drop("target", axis=1)
    y = df["target"]

    print("\nFeatures (X) and Target (y) split:")
    print("Features (X) sample:")
    print(X.head())
    print("Target (y) sample:")
    print(y.head())

    # 5. Encode Categorical Features
    categorical_cols = ["Gender", "Debtor", "Scholarship holder", "Tuition fees up to date"]
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print("\nDataset after encoding categorical features:")
    print(X.head())

    # 6. Train / Validation / Test Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print("\nTrain, Validation, and Test splits:")
    print("X_train sample:")
    print(X_train.head())
    print("X_val sample:")
    print(X_val.head())
    print("X_test sample:")
    print(X_test.head())

    # 7. Normalize Numerical Features
    numerical_cols = [
        "Age at enrollment",
        "Admission grade",
        "Previous qualification (grade)",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
        "Unemployment rate",
        "GDP",
        "Inflation rate"
    ]
    scaler = MinMaxScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    print("\nNormalized datasets:")
    print("X_train normalized sample:")
    print(X_train.head())
    print("X_val normalized sample:")
    print(X_val.head())
    print("X_test normalized sample:")
    print(X_test.head())

    return X_train, X_val, X_test, y_train, y_val, y_test