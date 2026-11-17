import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def bai_1():
    print("\n========== BAI 1: HOI QUY TUYEN TINH ==========")

    # Tao du lieu gia lap: y = 2X + 1 + nhieu ngau nhien.
    np.random.seed(42)
    X = np.arange(0, 50).reshape(-1, 1)
    noise = np.random.normal(0, 5, size=50)
    y = 2 * X.ravel() + 1 + noise

    model = LinearRegression()
    model.fit(X, y)

    print("He so (coefficient):", round(model.coef_[0], 4))
    print("He so chan (intercept):", round(model.intercept_, 4))
    print(f"Phuong trinh du doan: y = {model.coef_[0]:.4f} * X + {model.intercept_:.4f}")


def bai_2_va_3():
    print("\n========== BAI 2: DANH GIA MO HINH ==========")

    diabetes = load_diabetes()
    X = diabetes.data[:, 2].reshape(-1, 1)
    y = diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    print("\n========== BAI 3: TRUC QUAN HOA ==========")

    thu_tu = np.argsort(X_test[:, 0])
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test[:, 0], y_test, color="blue", alpha=0.7,
                label="Gia tri thuc te")
    plt.plot(X_test[thu_tu, 0], y_pred[thu_tu], color="red", linewidth=2,
             label="Duong hoi quy du doan")
    plt.title("Bai 3 - Hoi quy tuyen tinh tren du lieu Diabetes")
    plt.xlabel("Dac trung BMI (da chuan hoa trong bo du lieu)")
    plt.ylabel("Chi so tien trien benh (target)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def bai_4():
    print("\n========== BAI 4: QUY TRINH ML HOAN CHINH ==========")

    try:
        data = fetch_california_housing()
        ten_du_lieu = "California Housing"
    except (OSError, ValueError, RuntimeError) as loi:
        print("Khong tai duoc California Housing:", loi)
        print("Chuyen sang bo du lieu hoi quy Diabetes co san.")
        data = load_diabetes()
        ten_du_lieu = "Diabetes"

    X, y = data.data, data.target
    print("Du lieu:", ten_du_lieu)
    print("So mau:", X.shape[0], "| So dac trung:", X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE tren tap Test: {mse:.4f}")

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, color="green",
                label="Cac mau test")
    min_v = min(y_test.min(), y_pred.min())
    max_v = max(y_test.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--",
             label="Du doan hoan hao")
    plt.title(f"Bai 4 - Gia tri thuc te va du doan ({ten_du_lieu})")
    plt.xlabel("Gia tri thuc te")
    plt.ylabel("Gia tri du doan")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    bai_1()
    bai_2_va_3()
    bai_4()
    print("\nBAI 5: Giang vien giao them (de bai chua duoc cung cap).")
    print("\nHoan thanh Lab 3!")


if __name__ == "__main__":
    main()
