import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

### Data
df = pd.read_csv("AAPL History")
data = np.random.random((12,12))
# print(data)


### Side Bar
## Heading
st.sidebar.markdown("# 📊 Black-Scholes Model")
st.sidebar.markdown(":green[Created By:]")
st.sidebar.badge("Alexander Seo", color="grey")

## Inputs
current_asset_price = st.sidebar.number_input("Current Asset Price", min_value=0.00, value=100.00, step=1.00)
strike_price = st.sidebar.number_input("Strike Price", min_value=0.00, value=100.00, step=1.00)
maturity_time = st.sidebar.number_input("Time to Maturity (Years)", min_value=0.00, value=1.00, step=1.00)
volatility = st.sidebar.number_input("Volatility (σ)", min_value=0.00, value=.20, step=.10)
risk_free_interest_rate = st.sidebar.number_input("Risk-Free Interest Rate", min_value=0.00, value=.05, step=0.01)

## Heatmap Parametes
st.sidebar.badge("Heatmap Parameters", color="grey")
min_spot_price = st.sidebar.number_input("Min. Spot Price", min_value=0.00, value=80.00, step=1.00)
max_spot_price = st.sidebar.number_input("Max. Spot Price", min_value=0.00, value=120.00, step=1.00)
min_volatility = st.sidebar.slider("Min. Volatility for Heatmap", min_value=0.00, value=.10, max_value=1.00)
max_volatility = st.sidebar.slider("Max. Volatility for Heatmap", min_value=0.00, value=.30, max_value=1.00)


### Main Page
st.title("Black-Scholes Pricing Model")
var_df = pd.DataFrame(
    {"Current Asset Price": [current_asset_price], 
     "Strike Price": [strike_price],
     "Time to Maturity (Years)": [maturity_time],
     "Volatility (σ)": [volatility],
     "Risk-Free Interest Rate": [risk_free_interest_rate]
     }
)
st.write(var_df)
# st.chat_input("Variance")

