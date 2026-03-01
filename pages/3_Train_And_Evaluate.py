import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils import regression_metrics

st.title("🧠 Train & Evaluate Model")

if "df" not in st.session_state:
    st.warning("Please upload data first (Upload Data page).")
    st.stop()

df = st.session_state["df"]
x_col = st.session_state["x_col"]
y_col = st.session_state["y_col"]

st.subheader("Train/Test Split")
test_size = st.slider("Test size (%)", min_value=10, max_value=50, value=20, step=5) / 100.0
random_state = st.number_input("Random state (for reproducibility)", min_value=0, value=42, step=1)

X = df[[x_col]].values
y = df[y_col].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state)
)

st.write(f"Train rows: **{len(X_train)}** | Test rows: **{len(X_test)}**")

if st.button("Train Linear Regression ✅", type="primary"):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)

    st.session_state["model"] = model
    st.session_state["metrics"] = metrics
    st.session_state["train_info"] = {
        "coef": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "x_col": x_col,
        "y_col": y_col,
        "test_size": float(test_size),
        "random_state": int(random_state),
    }

    st.success("Model trained and evaluated ✅ Scroll down for results.")

if "model" not in st.session_state:
    st.info("Click **Train Linear Regression** to compute metrics.")
    st.stop()

model = st.session_state["model"]
metrics = st.session_state["metrics"]
train_info = st.session_state["train_info"]

st.subheader("Model Equation")
st.latex(rf"\hat{{y}} = {train_info['coef']:.4f} \cdot x + {train_info['intercept']:.4f}")

st.subheader("Metrics on Test Set")
st.write("Metrics: **R², RMSE, MSE, MAE, MAPE, Bias (MBE)**")
st.table(metrics)

st.subheader("Predictions vs Actuals (Test Set)")
# recreate the split for consistent plotting with settings
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=train_info["test_size"], random_state=train_info["random_state"]
)
y_pred = model.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(X_test, y_test, label="Actual")
ax.scatter(X_test, y_pred, label="Predicted")
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.set_title("Actual vs Predicted (Test Set)")
ax.legend()

st.pyplot(fig, clear_figure=True)
st.caption("Next: go to **Predict Salary** to try new experience values.")