import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    RidgeCV,
    LassoCV,
    ElasticNetCV,
    LogisticRegression,
)
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
import joblib
from pathlib import Path

# -------------------------------------------------
# Directories
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # .../eu
DATA_DIR = BASE_DIR                                 # CSVs are directly inside eu/
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# Helper to clean numbers
# -------------------------------------------------
def clean_number(s):
    if pd.isna(s):
        return np.nan
    s = str(s).replace(",", ".")
    s = re.sub(r"\s+", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan

def load_csv(path):
    try:
        return pd.read_csv(path, sep=";", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=";", encoding="latin1")

# -------------------------------------------------
# Load datasets
# -------------------------------------------------
products = load_csv(DATA_DIR / "Products.csv")
orders = load_csv(DATA_DIR / "Orders.csv")
customers = load_csv(DATA_DIR / "Customers.csv")
location = load_csv(DATA_DIR / "Location.csv")

print("✅ Datasets loaded:")
print("Products:", products.shape)
print("Orders:", orders.shape)
print("Customers:", customers.shape)
print("Location:", location.shape)

# -------------------------------------------------
# Clean numbers in orders
# -------------------------------------------------
orders["Sales"] = orders["Sales"].apply(clean_number)
orders["Profit"] = orders["Profit"].apply(clean_number)
orders["Quantity"] = pd.to_numeric(orders["Quantity"], errors="coerce")
orders["Discount"] = pd.to_numeric(orders["Discount"], errors="coerce")

# Parse dates
orders["Order Date"] = pd.to_datetime(orders["Order Date"], errors="coerce")

# -------------------------------------------------
# Merge datasets
# -------------------------------------------------
df = (
    orders.merge(products, on="Product ID", how="left")
    .merge(customers, on="Customer ID", how="left")
    .merge(location, on="Postal Code", how="left")
)

# -------------------------------------------------
# Feature engineering
# -------------------------------------------------
df["Profitable"] = (df["Profit"] > 0).astype(int)
df["Year"] = df["Order Date"].dt.year
df["LogSales"] = np.log1p(df["Sales"].fillna(0))

# -------------------------------------------------
# Regression (predicting log sales)
# -------------------------------------------------
X = df[["Quantity", "Discount"]].fillna(0)
y = df["LogSales"].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# OLS
ols = sm.OLS(y_train, sm.add_constant(X_train)).fit()
with open(OUTPUT_DIR / "ols_summary.txt", "w") as f:
    f.write(ols.summary().as_text())

# Linear Regression
lr = LinearRegression().fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print("Linear R²:", r2_score(y_test, pred_lr))
joblib.dump(lr, OUTPUT_DIR / "linear_model.joblib")

# Ridge Regression
ridge = RidgeCV(alphas=np.logspace(-3, 3, 50)).fit(X_train, y_train)
print("Best alpha Ridge:", ridge.alpha_)
joblib.dump(ridge, OUTPUT_DIR / "ridge_model.joblib")

# Lasso Regression
lasso = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X_train, y_train)
print("Best alpha Lasso:", lasso.alpha_)
joblib.dump(lasso, OUTPUT_DIR / "lasso_model.joblib")

# ElasticNet
enet = ElasticNetCV(cv=5, random_state=42).fit(X_train, y_train)
print("ElasticNet alpha:", enet.alpha_, "l1_ratio:", enet.l1_ratio_)
joblib.dump(enet, OUTPUT_DIR / "elasticnet_model.joblib")

# -------------------------------------------------
# Logistic Regression (Profitable or not)
# -------------------------------------------------
X_cls = df[["Quantity", "Discount"]].fillna(0)
y_cls = df["Profitable"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=1000).fit(X_train_c, y_train_c)

acc = accuracy_score(y_test_c, clf.predict(X_test_c))
roc = roc_auc_score(y_test_c, clf.predict_proba(X_test_c)[:, 1])
print("Logistic Accuracy:", acc, "ROC AUC:", roc)

joblib.dump(clf, OUTPUT_DIR / "logistic_model.joblib")

# -------------------------------------------------
# Visualization
# -------------------------------------------------

# 1. Scatter: actual vs predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, pred_lr, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Log(Sales)")
plt.ylabel("Predicted Log(Sales)")
plt.title("Linear Regression: Actual vs Predicted")
plt.savefig(OUTPUT_DIR / "linear_regression_scatter.png")
plt.close()

# 2. Residual plot
residuals = y_test - pred_lr
plt.figure(figsize=(6,4))
plt.scatter(pred_lr, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Log(Sales)")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig(OUTPUT_DIR / "residuals.png")
plt.close()

# 3. Coefficients bar plot (Linear model)
coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": lr.coef_})
plt.figure(figsize=(6,4))
plt.bar(coef_df["Feature"], coef_df["Coefficient"])
plt.title("Linear Regression Coefficients")
plt.savefig(OUTPUT_DIR / "coefficients.png")
plt.close()

# 4. Logistic ROC curve
y_score = clf.predict_proba(X_test_c)[:, 1]
fpr, tpr, _ = roc_curve(y_test_c, y_score)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.legend(loc="lower right")
plt.savefig(OUTPUT_DIR / "roc_curve.png")
plt.close()

print("✅ All models trained, saved, and graphs exported to:", OUTPUT_DIR)
