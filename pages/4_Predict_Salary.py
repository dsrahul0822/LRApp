import streamlit as st
import numpy as np

st.title("💰 Predict Salary")

if "df" not in st.session_state:
    st.warning("Please upload data first (Upload Data page).")
    st.stop()

if "model" not in st.session_state:
    st.warning("Please train the model first (Train & Evaluate page).")
    st.stop()

df = st.session_state["df"]
x_col = st.session_state["x_col"]
y_col = st.session_state["y_col"]
model = st.session_state["model"]

st.write("Enter a new experience value. The model will predict a salary.")

min_x = float(df[x_col].min())
max_x = float(df[x_col].max())

exp = st.number_input(
    f"{x_col} (years)",
    min_value=0.0,
    value=float(round((min_x + max_x) / 2, 1)),
    step=0.1,
)

if st.button("Predict 💡", type="primary"):
    pred = float(model.predict(np.array([[exp]], dtype=float))[0])
    st.success(f"Predicted **{y_col}** for **{x_col} = {exp}** is: **{pred:,.2f}**")
    st.caption("For real offers, consider role level, location, market bands, budget, etc.")