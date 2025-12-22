import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv("AAPL History")

data = np.random.random((12,12))
# print(data)

st.title("Interactive Call and Put Model (Heatmap Included)")
st.line_chart(df["Close"])
st.chat_input("Variance")
