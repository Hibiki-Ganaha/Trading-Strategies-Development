import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv("AAPL History")

data = np.random.random((12,12))
# print(data)

## Side Bar
st.sidebar.markdown("# 📊 Black-Scholes Model")
st.sidebar.markdown(":green[Created By:]")
st.sidebar.markdown(
    "st.sidebar.image('linkedinlogo.png',width=30) st.badge('Alexander Seo')"
)


st.title("Black-Scholes Pricing Model")
st.write(df.head())
# st.chat_input("Variance")

