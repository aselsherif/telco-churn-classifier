
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="My First App",
    page_icon="📊",
    layout="wide",
)

st.title("My First Streamlit App")
st.write("Welcome to my data application!")

name = st.text_input("Enter your name")

if st.button("Submit"):
    if name:
        st.success(f"Welcome, {name}!")
    else:
        st.warning("Please enter your name.")

st.subheader("Sample Data")

df = pd.DataFrame({
    "Customer": ["Ahmed", "Sara", "Omar"],
    "Age": [25, 30, 28],
    "Churn": ["No", "Yes", "No"],
})

st.dataframe(df, use_container_width=True)
