import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_digits
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ==================================================
# BAI 1: LAM SACH DU LIEU
# ==================================================
print('========== BAI 1: LAM SACH DU LIEU ==========')
data = {
    'Tuoi': [20, 25, np.nan, 30, 35, 40, 45, 50, 55, 28, 32, 38, 42, 47, 200],
    'Thu_nhap': [10, 15, 20, np.nan, 30, 35, 40, 45, 50, 25, 32, 38, 42, 47, 500]
}
df = pd.DataFrame(data)
print('\nDu lieu ban dau:')
print(df)

df = df.fillna(df.mean())
print('\nDu lieu sau khi dien gia tri thieu:')
print(df)

# Xac dinh outlier theo quy tac |x - mean| > 3 * std
for col in df.select_dtypes(include=np.number).columns:
    mean = df[col].mean()
    std = df[col].std()
    outlier_mask = np.abs(df[col] - mean) > 3 * std
    print(f'\nOutlier cot {col}:', df.loc[outlier_mask, col].tolist())
    df.loc[outlier_mask, col] = df[col].median()

print('\nDu lieu sau khi xu ly outlier:')
print(df)

# ==================================================
# BAI 2: CHUAN HOA DU LIEU (SCALING)
# ==================================================
print('\n========== BAI 2: CHUAN HOA DU LIEU ==========')
data = np.array([
    [20, 10], [25, 20], [30, 30], [35, 40],
    [40, 50], [45, 60], [50, 80], [60, 100]
])
print('\nDu lieu ban dau (Tuoi, Thu nhap):')
print(data)

standard_scaler = StandardScaler()
data_standard = standard_scaler.fit_transform(data)
print('\nDu lieu sau StandardScaler:')
print(data_standard)

minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(data)
print('\nDu lieu sau MinMaxScaler:')
print(data_minmax)

# ==================================================
# BAI 3: CHIA TAP DU LIEU TRAIN / TEST
# ==================================================
print('\n========== BAI 3: CHIA TRAIN / TEST ==========')
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
print('\nKich thuoc du lieu goc:')
print('X:', X.shape)
print('y:', y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print('\nKich thuoc tap Train:')
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('\nKich thuoc tap Test:')
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)

# ==================================================
# BAI 4: QUY TRINH ML - TIEN XU LY DU LIEU
# ==================================================
print('\n========== BAI 4: PIPELINE MACHINE LEARNING ==========')
digits = load_digits()
X = digits.data
y = digits.target
print('\nKich thuoc du lieu digits:')
print('X:', X.shape)
print('y:', y.shape)
print('\nGia tri pixel nho nhat:', X.min())
print('Gia tri pixel lon nhat:', X.max())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print('\nDo chinh xac cua mo hinh:')
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nBao cao danh gia mo hinh:')
print(classification_report(y_test, y_pred))
print('\n========== HOAN THANH LAB 2 ==========')
