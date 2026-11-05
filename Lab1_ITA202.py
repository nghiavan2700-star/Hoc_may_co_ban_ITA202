# LAB 1 - TONG QUAN HOC MAY VA CAI DAT MOI TRUONG

# BAI 1: NUMPY VA PANDAS
import numpy as np
import pandas as pd

print("===== BAI 1: NUMPY VA PANDAS =====")

# Tao mang Numpy gom cac so tu 1 den 10
mang = np.arange(1, 11)
print("Mang Numpy tu 1 den 10:")
print(mang)

# Tao DataFrame thong tin 5 sinh vien
data = {
    "Ho ten": ["An", "Binh", "Chi", "Dung", "Hoa"],
    "Tuoi": [19, 20, 19, 21, 20],
    "Diem toan": [8.0, 7.5, 9.0, 6.5, 8.5]
}

df = pd.DataFrame(data)

print("\nBang thong tin 5 sinh vien:")
print(df)

print("\nThong tin tong quan cua DataFrame:")
df.info()

# Tinh diem trung binh mon Toan
diem_trung_binh = df["Diem toan"].mean()
print("\nDiem trung binh mon Toan:", diem_trung_binh)

# BAI 2: PHAN LOAI BAI TOAN MACHINE LEARNING

print("\n===== BAI 2: PHAN LOAI MACHINE LEARNING =====")

print("Da ghi phan loai cac vi du bang comment trong code.")

# BAI 3: MO HINH DON GIAN VOI SCIKIT-LEARN

print("\n===== BAI 3: MO HINH DON GIAN =====")

from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

# Buoc 1: Load du lieu Iris
iris = load_iris()
X = iris.data      # Du lieu dau vao
y = iris.target    # Nhan cua tung mau

model = LinearRegression()

model.fit(X, y)

ket_qua_du_doan = model.predict(X[:5])

print("Ket qua du doan 5 mau dau tien:")
print(ket_qua_du_doan)

print("Nhan that cua 5 mau dau tien:")
print(y[:5])

# BAI 4: QUY TRINH ML - THU THAP VA KIEM TRA DU LIEU

print("\n===== BAI 4: KIEM TRA DU LIEU DIGITS =====")

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Tai tap du lieu digits
digits = load_digits()

X_digits = digits.data
y_digits = digits.target

# In hinh dang cua du lieu va nhan
print("Shape cua du lieu:", X_digits.shape)
print("Shape cua nhan:", y_digits.shape)

# In nhan cua mau dau tien
print("Nhan cua mau dau tien:", y_digits[0])

# Hien thi anh dau tien
plt.imshow(digits.images[0], cmap="gray")
plt.title("Mau dau tien - Nhan: " + str(y_digits[0]))
plt.axis("off")
plt.show()
