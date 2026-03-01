import streamlit as st
import pandas as pd
from utils import validate_numeric_series

st.title("📁 Upload Data (CSV)")
st.write("Upload your CSV file. We'll store it in session memory for the next pages.")

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
st.caption("Tip: For your demo, upload `Salary_Data.csv` (YearsExperience, Salary).")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        st.stop()

    st.subheader("Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("Select columns")
    cols = list(df.columns)

    default_x = "YearsExperience" if "YearsExperience" in cols else cols[0]
    default_y = "Salary" if "Salary" in cols else (cols[1] if len(cols) > 1 else cols[0])

    x_col = st.selectbox("Feature (X): experience column", cols, index=cols.index(default_x))
    y_col = st.selectbox("Target (y): salary column", cols, index=cols.index(default_y))

    clean = df[[x_col, y_col]].copy()
    clean[x_col] = validate_numeric_series(clean[x_col])
    clean[y_col] = validate_numeric_series(clean[y_col])
    clean = clean.dropna()

    st.write("After converting to numeric and dropping missing values:")
    st.dataframe(clean, use_container_width=True)

    if len(clean) < 5:
        st.warning("Not enough clean rows after conversion. Please check your columns.")
        st.stop()

    st.session_state["df_raw"] = df
    st.session_state["df"] = clean
    st.session_state["x_col"] = x_col
    st.session_state["y_col"] = y_col

    # reset training artifacts if user uploads again
    st.session_state.pop("model", None)
    st.session_state.pop("metrics", None)
    st.session_state.pop("train_info", None)

    st.success("Saved ✅ Now go to **Visualize** page.")
else:
    st.info("Upload a CSV to continue.")