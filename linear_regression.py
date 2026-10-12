import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # remove this line if you want plt.show() windows
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from solubility_detector_class_df import MolecularFeatures

CSV_PATH = "delaney-processed.csv"
TARGET = sys.argv[1] if len(sys.argv) > 1 else "measured"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# =========================================================
# BUILD DATASET
# =========================================================
features = MolecularFeatures(CSV_PATH, target=TARGET)
df = features.get_full_dataset()

y = df["solubility"].values
X_desc = df[features.descriptor_cols].values
X_fp = df[features.fingerprint_cols].values

feature_sets = {
    "Descriptors": X_desc,
    "Morgan": X_fp,
    "Descriptors + Morgan": np.hstack((X_desc, X_fp)),
}

models = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

# One shared split (same rows for every model/feature set -> fair comparison)
idx_train, idx_test = train_test_split(
    np.arange(len(y)), test_size=TEST_SIZE, random_state=RANDOM_STATE
)
y_train, y_test = y[idx_train], y[idx_test]


def metrics(y_true, y_pred):
    return (
        r2_score(y_true, y_pred),
        root_mean_squared_error(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
    )


# =========================================================
# RUN EVERYTHING
# =========================================================
rows, preds_store = [], {}
for fs_name, X in feature_sets.items():
    for m_name, make_model in models.items():
        model = make_model().fit(X[idx_train], y_train)
        p_train = model.predict(X[idx_train])
        p_test = model.predict(X[idx_test])
        tr, te = metrics(y_train, p_train), metrics(y_test, p_test)
        rows.append({
            "Features": fs_name, "Model": m_name,
            "Train R2": tr[0], "Test R2": te[0],
            "Test RMSE": te[1], "Test MAE": te[2],
        })
        preds_store[(fs_name, m_name)] = p_test

results = pd.DataFrame(rows)
print(f"\nTarget: {TARGET}   (n_train={len(y_train)}, n_test={len(y_test)})")
print(results.round(3).to_string(index=False))

# =========================================================
# PLOTS: predicted vs actual, 3 feature sets x 2 models
# =========================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
lims = [y_test.min(), y_test.max()]
for col, fs_name in enumerate(feature_sets):
    for row, m_name in enumerate(models):
        ax = axes[row, col]
        ax.scatter(y_test, preds_store[(fs_name, m_name)], alpha=0.6)
        ax.plot(lims, lims, "k--")
        ax.set_title(f"{m_name} - {fs_name}")
        ax.set_xlabel("Actual Solubility")
        ax.set_ylabel("Predicted Solubility")
plt.tight_layout()
plt.savefig(f"comparison_{TARGET}.png", dpi=120)
