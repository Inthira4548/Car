# ============================================================
# LAB : Classification ด้วย ANN
# Dataset : car_purchasing.csv
# Train 70% / Test 30%
# เปรียบเทียบ ANN, Naive Bayes, Decision Tree, KNN และ SVM
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. โหลด Dataset
# ============================================================

df = pd.read_csv(
    "car_purchasing.csv",
    encoding="latin1"
)

print("ขนาด Dataset :", df.shape)
print("\nข้อมูล 5 แถวแรก")
print(df.head())

print("\nข้อมูลแต่ละ Column")
print(df.columns)


# ============================================================
# 2. ตรวจสอบข้อมูล
# ============================================================

print("\nจำนวน Missing Value")
print(df.isnull().sum())

print("\nจำนวนข้อมูลซ้ำ :", df.duplicated().sum())


# ============================================================
# 3. สร้าง Class จาก car purchase amount
# ============================================================

# ใช้ Median เป็นจุดแบ่ง
median_price = df["car purchase amount"].median()

print("\nMedian ของราคาซื้อรถ =", median_price)

# 0 = ราคาซื้อรถต่ำ
# 1 = ราคาซื้อรถสูง

df["purchase_class"] = np.where(
    df["car purchase amount"] >= median_price,
    1,
    0
)

print("\nจำนวนข้อมูลแต่ละ Class")
print(df["purchase_class"].value_counts())


# ============================================================
# 4. เลือก Features
# ============================================================

X = df[
    [
        "gender",
        "age",
        "annual Salary",
        "credit card debt",
        "net worth"
    ]
]

y = df["purchase_class"]


# ============================================================
# 5. แบ่งข้อมูล Train 70% / Test 30%
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nจำนวน Train :", len(X_train))
print("จำนวน Test  :", len(X_test))


# ============================================================
# 6. Standardization
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 7. สร้าง Models
# ============================================================

models = {

    # ANN
    "ANN": MLPClassifier(
        hidden_layer_sizes=(10, 10, 10,10,10,10,10,10,10,10),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42
    ),

    # Naive Bayes
    "Naive Bayes": GaussianNB(),

    # Decision Tree
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    # KNN
    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    # SVM
    "SVM": SVC(
        kernel="rbf",
        random_state=42
    )
}


# ============================================================
# 8. Train + Test ทุก Model
# ============================================================

results = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print("Model :", name)
    print("=" * 60)

    # Model ที่ใช้ข้อมูล Standardized
    if name in ["ANN", "KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # เก็บผลลัพธ์
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F-measure": f1
    })

    print("Accuracy  :", round(accuracy, 4))
    print("Precision :", round(precision, 4))
    print("Recall    :", round(recall, 4))
    print("F-measure :", round(f1, 4))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "ราคาต่ำ",
                "ราคาสูง"
            ],
            zero_division=0
        )
    )


# ============================================================
# 9. สรุปผลทุก Model
# ============================================================

result_df = pd.DataFrame(results)

print("\n")
print("=" * 80)
print("สรุปผลการทดลอง")
print("=" * 80)

print(
    result_df.to_string(
        index=False
    )
)


# ============================================================
# 10. แสดงค่าเป็นเปอร์เซ็นต์
# ============================================================

result_percent = result_df.copy()

for col in [
    "Accuracy",
    "Precision",
    "Recall",
    "F-measure"
]:
    result_percent[col] = (
        result_percent[col] * 100
    ).round(2)

print("\nผลลัพธ์เป็นเปอร์เซ็นต์")
print(
    result_percent.to_string(
        index=False
    )
)