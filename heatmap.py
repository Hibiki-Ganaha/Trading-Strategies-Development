import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

### Data
df = pd.read_csv("AAPL History")
df.index = df.index.astype("datetime64[s, UTC-05:00]")
data = np.random.random((12,12))
# print(data)


### Side Bar
## Heading
st.sidebar.title("📊 Black-Scholes Model")
st.sidebar.markdown(":green[Created By:]")
col1, col2 = st.sidebar.columns(2, gap=None)
col1.image("linkedinlogo.png", width=30)
col2.page_link("https://www.linkedin.com/in/alexanderminhseo/", label="Alexander Seo")

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
## Header
st.title("Black-Scholes Pricing Model")
var_df = pd.DataFrame(
    {"Current Asset Price ($)": [current_asset_price], 
     "Strike Price ($)": [strike_price],
     "Time to Maturity (Years)": [maturity_time],
     "Volatility (σ)": [volatility],
     "Risk-Free Interest Rate": [risk_free_interest_rate]
     }
)
st.write(var_df)

## Using the Black-Scholes Model Formula: 
d1 = (np.log(current_asset_price/strike_price) + (risk_free_interest_rate + volatility**2 / 2) * maturity_time) / (volatility * np.sqrt(maturity_time))
# print(d1)
d2 = d1 - volatility * np.sqrt(maturity_time)
# print(d2)
call_value = current_asset_price * ss.norm.cdf(d1) - strike_price * np.exp(-risk_free_interest_rate * maturity_time) * ss.norm.cdf(d2)
# print(call_value)
put_value = strike_price * np.exp(-risk_free_interest_rate * maturity_time) * ss.norm.cdf(-d2) - current_asset_price * ss.norm.cdf(-d1)
col1, col2 = st.columns(2, gap="small", border=True)
col1.metric("CALL Value", "$" + str(round(call_value, 2)))
col2.metric("PUT Value", "$" + str(round(put_value, 2)))

## Interactive Heatmap
np.linspace(min_volatility, max_volatility, 10)
np.linspace(min_spot_price, max_spot_price, 10)

sns.heatmap(np.random.rand(10,10))
plt.show()