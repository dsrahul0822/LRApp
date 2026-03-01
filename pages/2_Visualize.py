import streamlit as st
import matplotlib.pyplot as plt

st.title("📊 Visualize Relationship")

if "df" not in st.session_state:
    st.warning("Please upload data first (Upload Data page).")
    st.stop()

df = st.session_state["df"]
x_col = st.session_state["x_col"]
y_col = st.session_state["y_col"]

st.write("Scatter plot to check relationship between experience and salary.")

fig, ax = plt.subplots()
ax.scatter(df[x_col], df[y_col])
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.set_title(f"{y_col} vs {x_col}")

st.pyplot(fig, clear_figure=True)
st.caption("If points show an upward straight trend, Linear Regression is a good starting point.")