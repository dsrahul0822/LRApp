import streamlit as st

st.set_page_config(
    page_title="Salary vs Experience • Linear Regression Demo",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Salary vs Experience — Linear Regression (Streamlit Demo)")
st.write(
    """This mini app has 4 pages:

1) **Upload Data** (CSV)  
2) **Visualize** (scatter plot)  
3) **Train & Evaluate** (train/test split + metrics)  
4) **Predict Salary** (enter new experience)  

Use the left sidebar to navigate."""
)

with st.expander("What you need"):
    st.markdown(
        """- A CSV file with columns **YearsExperience** and **Salary** (or choose columns on Upload page).
- Internet is not required.
- This is a simple demo app: one model (Linear Regression), one dataset."""
    )

st.info("Start with **Upload Data** from the sidebar 👈")