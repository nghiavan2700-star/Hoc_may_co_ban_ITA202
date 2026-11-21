# LAB 4 - ITA202 - PHAN LOAI BANG LOGISTIC REGRESSION

from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


# ==================== BAI 1 ====================
print('========== BAI 1: LOGISTIC REGRESSION =========')
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train_scaled, y_train)
print('So mau train:', len(X_train))
print('So mau test :', len(X_test))
print('Huan luyen mo hinh thanh cong!')


# ==================== BAI 2 ====================
print('\n========== BAI 2: DANH GIA MO HINH =========')
y_pred = model.predict(X_test_scaled)
print('Accuracy :', round(accuracy_score(y_test, y_pred), 4))
print('Precision:', round(precision_score(y_test, y_pred), 4))
print('Recall   :', round(recall_score(y_test, y_pred), 4))
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=cancer.target_names))


# ==================== BAI 3 ====================
print('========== BAI 3: CONFUSION MATRIX =========')
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print('Ma tran nham lan:\n', cm)

# Theo quy uoc nhan duong tinh la 1:
# TN: thuc te = 0, du doan = 0 (dung am tinh)
# FP: thuc te = 0, du doan = 1 (sai duong tinh)
# FN: thuc te = 1, du doan = 0 (sai am tinh)
# TP: thuc te = 1, du doan = 1 (dung duong tinh)
tn, fp, fn, tp = cm.ravel()
print('TN =', tn, '| FP =', fp, '| FN =', fn, '| TP =', tp)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cancer.target_names)
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Bai 3 - Breast Cancer Confusion Matrix')
fig.tight_layout()
fig.savefig('Lab4_Bai3_ConfusionMatrix.png', dpi=150)
print('Da luu bieu do: Lab4_Bai3_ConfusionMatrix.png')
plt.close(fig)


# ==================== BAI 4 ====================
print('\n========== BAI 4: LOGISTIC REGRESSION - DIGITS =========')
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_digits, y_digits, test_size=0.2, random_state=42, stratify=y_digits
)
scaler_d = StandardScaler()
X_train_d = scaler_d.fit_transform(X_train_d)
X_test_d = scaler_d.transform(X_test_d)

model_d = LogisticRegression(max_iter=3000, random_state=42)
model_d.fit(X_train_d, y_train_d)
y_pred_d = model_d.predict(X_test_d)

print('Accuracy (Digits):', round(accuracy_score(y_test_d, y_pred_d), 4))
print('\nClassification Report (Digits):')
print(classification_report(y_test_d, y_pred_d, zero_division=0))

wrong_indices = [i for i in range(len(y_test_d)) if y_test_d[i] != y_pred_d[i]]
print('Tong so mau du doan sai:', len(wrong_indices))
print('5 mau du doan sai dau tien (nhan thuc te -> nhan du doan):')
for stt, i in enumerate(wrong_indices[:5], start=1):
    print(f'  Mau {stt}: {y_test_d[i]} -> {y_pred_d[i]}')

if len(wrong_indices) < 5:
    print('Luu y: Mo hinh co it hon 5 mau du doan sai.')

print('\n========== HOAN THANH LAB 4 =========')
