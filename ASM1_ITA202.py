from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

from Lab2_ITA202 import X_train
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def main():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'target']
    features = [c for c in df if c != 'target']
    df.to_csv(DATA / 'iris_raw.csv', index=False)

    print('ITA202 - ASSIGNMENT 1: IRIS DATA PREPROCESSING')
    print('Dataset: Iris (scikit-learn / Fisher 1936)')
    print('Original shape:', df.shape)
    print('Target classes:', iris.target_names.tolist())
    print('Missing values (original):', int(df.isna().sum().sum()))
    print('Duplicate rows (original):', int(df.duplicated().sum()))
    print('\nDATAFRAME INFO')
    df.info()
    print('\nDESCRIBE')
    print(df.describe().round(2).to_string())

    # Stratified split BEFORE learning any cleaning/scaling statistics (avoid leakage).
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df['target'], test_size=0.20, random_state=42,
        stratify=df['target']
    )
    # Median imputation is fit using training rows only.
    imputer = SimpleImputer(strategy='median')
    train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=features, index=X_train.index)
    test_imp = pd.DataFrame(imputer.transform(X_test), columns=features, index=X_test.index)

    # IQR caps learned from train only; treat extreme readings by winsorization.
    q1 = train_imp.quantile(0.25)
    q3 = train_imp.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    train_outliers = int(((train_imp < low) | (train_imp > high)).any(axis=1).sum())
    test_outliers = int(((test_imp < low) | (test_imp > high)).any(axis=1).sum())
    train_clip = train_imp.clip(lower=low, upper=high, axis=1)
    test_clip = test_imp.clip(lower=low, upper=high, axis=1)

    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_clip), columns=features, index=X_train.index)
    test_scaled = pd.DataFrame(scaler.transform(test_clip), columns=features, index=X_test.index)

    train_result = train_scaled.assign(target=y_train, split='train').sort_index()
    test_result = test_scaled.assign(target=y_test, split='test').sort_index()
    train_result.to_csv(DATA / 'iris_train_processed.csv', index=False)
    test_result.to_csv(DATA / 'iris_test_processed.csv', index=False)
    pd.concat([train_result, test_result]).sort_index().to_csv(DATA / 'iris_processed_all.csv', index=False)

    # First basic baseline, as specified in the full assignment PDF.
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(train_scaled, y_train)
    predictions = model.predict(test_scaled)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=iris.target_names, zero_division=0)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Iris - Logistic Regression (20% test set)')
    plt.tight_layout()
    fig.savefig(RESULTS / 'confusion_matrix.png', dpi=170)
    plt.close(fig)

    metrics = {
        'dataset': 'Iris - sklearn.datasets.load_iris',
        'original_rows': len(df), 'features': len(features),
        'original_missing_values': int(df.isna().sum().sum()),
        'original_duplicate_rows': int(df.duplicated().sum()),
        'train_rows': len(X_train), 'test_rows': len(X_test),
        'train_rows_with_iqr_outliers': train_outliers,
        'test_rows_with_iqr_outliers': test_outliers,
        'missing_after_preprocessing': int(train_scaled.isna().sum().sum() + test_scaled.isna().sum().sum()),
        'baseline_test_accuracy': round(float(acc), 6),
        'confusion_matrix': cm.tolist(), 'classification_report': report,
        'random_state': 42
    }
    (RESULTS / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')

    print('\nPREPROCESSING RESULTS')
    print(f'Train/Test rows: {len(X_train)} / {len(X_test)} (80% / 20%)')
    print('Training rows with IQR outliers:', train_outliers)
    print('Test rows with IQR outliers:', test_outliers)
    print('Missing after preprocessing:', metrics['missing_after_preprocessing'])
    print('Train scaled mean:', np.round(train_scaled.mean().values, 3))
    print('Train scaled std:', np.round(train_scaled.std(ddof=0).values, 3))
    print('Files saved: iris_raw.csv, iris_train_processed.csv, iris_test_processed.csv, iris_processed_all.csv')
    print('\nBASELINE LOGISTIC REGRESSION')
    print(f'Test accuracy: {acc:.4f}')
    print('Confusion matrix:\n', cm)
    print(report)
    print('Saved: results/confusion_matrix.png, results/metrics.json')


if __name__ == '__main__':
    main()
