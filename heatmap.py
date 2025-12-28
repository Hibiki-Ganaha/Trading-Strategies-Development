import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import requests
import time


### Data
df = pd.read_csv("AAPL History")
df.index = df.index.astype("datetime64[s, UTC-05:00]")
data = np.random.random((12,12))
# print(data)


### Side Bar
## Heading
st.set_page_config(layout="wide")
st.sidebar.title("📊 Black-Scholes Model")
st.sidebar.markdown(":green[Created By:]")
# col1, col2 = st.sidebar.columns(2, gap=None)
# col1.image("linkedinlogo.png", width=30)
# col2.page_link("https://www.linkedin.com/in/alexanderminhseo/", label="Alexander Seo")
st.sidebar.markdown(f"<a href='{'https://www.linkedin.com/in/alexanderminhseo/'}' target='_blank' style='text-decoration: none; color: inherit;'><img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width='20' height='20' style='vertical-align: middle; margin-right: 10px;'>`Alexander Seo`</a>", unsafe_allow_html=True)


## Inputs
current_asset_price = st.sidebar.number_input("Current Asset Price", min_value=0.00, value=100.00, step=1.00)
strike_price = st.sidebar.number_input("Strike Price", min_value=0.00, value=100.00, step=1.00)
maturity_time = st.sidebar.number_input("Time to Maturity (Years)", min_value=0.00, value=1.00, step=.10)
volatility = st.sidebar.number_input("Volatility (σ)", min_value=0.00, value=.20, step=.05)
risk_free_interest_rate = st.sidebar.number_input("Risk-Free Interest Rate", min_value=0.00, value=.05, step=0.01)

## Heatmap Parameters
st.sidebar.badge("Heatmap Parameters", color="grey")
min_spot_price = st.sidebar.number_input("Min. Spot Price", min_value=0.00, value=80.00, step=1.00)
max_spot_price = st.sidebar.number_input("Max. Spot Price", min_value=0.00, value=120.00, step=1.00)
min_volatility = st.sidebar.slider("Min. Volatility for Heatmap", min_value=0.01, value=.10, max_value=1.00)
max_volatility = st.sidebar.slider("Max. Volatility for Heatmap", min_value=0.01, value=.50, max_value=1.00)


### Main Page
## Header
st.title("Black-Scholes Pricing Model")
# var_df = pd.DataFrame(
#     {"Current Asset Price ($)": [current_asset_price], 
#      "Strike Price ($)": [strike_price],
#      "Time to Maturity (Years)": [maturity_time],
#      "Volatility (σ)": [volatility],
#      "Risk-Free Interest Rate": [risk_free_interest_rate]
#      }
# )
# st.write(var_df)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Asset Price ($)", round(current_asset_price, 2))
col2.metric("Strike Price ($)", round(strike_price, 2))
col3.metric("Time to Maturity (Years)", round(maturity_time, 2))
col4.metric("Volatility (σ)", round(volatility, 2))
col5.metric("Risk-Free Interest Rate", round(risk_free_interest_rate, 2))

## Using the Black-Scholes Model Formula:
def d1(current_asset_price, strike_price, risk_free_interest_rate, volatility, maturity_time):
    return (np.log(current_asset_price/strike_price) + (risk_free_interest_rate + volatility**2 / 2) * maturity_time) / (volatility * np.sqrt(maturity_time))

# print(d1(current_asset_price, strike_price, risk_free_interest_rate, volatility, maturity_time))

def d2(current_asset_price, strike_price, risk_free_interest_rate, volatility, maturity_time):
    return d1(current_asset_price, strike_price, risk_free_interest_rate, volatility, maturity_time) - volatility * np.sqrt(maturity_time)

# print(d2(current_asset_price, strike_price, risk_free_interest_rate, volatility, maturity_time))

def call_value(current_asset_price, strike_price, risk_free_interest_rate, volatility, maturity_time):
    return current_asset_price * ss.norm.cdf(d1(current_asset_price, strike_price, risk_free_interest_rate, volatility, maturity_time)) - strike_price * np.exp(-risk_free_interest_rate * maturity_time) * ss.norm.cdf(d2(current_asset_price, strike_price, risk_free_interest_rate, volatility, maturity_time))

# print(call_value(current_asset_price, strike_price, risk_free_interest_rate, volatility, maturity_time))

def put_value(current_asset_price, strike_price, risk_free_interest_rate, volatility, maturity_time):
    return strike_price * np.exp(-risk_free_interest_rate * maturity_time) * ss.norm.cdf(-d2(current_asset_price, strike_price, risk_free_interest_rate, volatility, maturity_time)) - current_asset_price * ss.norm.cdf(-d1(current_asset_price, strike_price, risk_free_interest_rate, volatility, maturity_time))

# print(put_value(current_asset_price, strike_price, risk_free_interest_rate, volatility, maturity_time))

col1, col2 = st.columns(2, gap="small", border=True)
col1.metric("CALL Value", "$" + str(round(call_value(current_asset_price, strike_price, risk_free_interest_rate, volatility, maturity_time), 2)))
col2.metric("PUT Value", "$" + str(round(put_value(current_asset_price, strike_price, risk_free_interest_rate, volatility, maturity_time), 2)))

st.header("Options Price - Interactive Heatmap")
st.info('Explore how option prices fluctuate with varying "Spot Prices and Volatility" levels using interactive heatmap parameters, all while maintaining a constant "Strike Price".')

## Interactive Heatmap
vol = np.round(np.linspace(min_volatility, max_volatility, 10), 2)
spot = np.round(np.linspace(min_spot_price, max_spot_price, 10), 2)
# print(vol)
# print(spot)

# call_vals = current_asset_price * ss.norm.cdf(d1(spot[np.newaxis, :], strike_price, risk_free_interest_rate, vol[:, np.newaxis], maturity_time)) - strike_price * np.exp(-risk_free_interest_rate * maturity_time) * ss.norm.cdf(d2(spot[np.newaxis, :], strike_price, risk_free_interest_rate, vol[:, np.newaxis], maturity_time))
# call_vals =  spot[np.newaxis, :] + vol[:, np.newaxis]
# print(ss.norm.cdf(d1(spot[np.newaxis, :], strike_price, risk_free_interest_rate, vol[:, np.newaxis], maturity_time)))
# print(ss.norm.cdf(d2(spot[np.newaxis, :], strike_price, risk_free_interest_rate, vol[:, np.newaxis], maturity_time)))
# print( ss.norm.cdf(d1(spot[np.newaxis, :], strike_price, risk_free_interest_rate, vol[:, np.newaxis], maturity_time)) - ss.norm.cdf(d2(spot[np.newaxis, :], strike_price, risk_free_interest_rate, vol[:, np.newaxis], maturity_time)) )
# print(d1(spot[np.newaxis, :], strike_price, risk_free_interest_rate, vol[:, np.newaxis], maturity_time))
# print(call_vals)
# print(d1(current_asset_price, spot[:, np.newaxis], risk_free_interest_rate, vol[np.newaxis:, ], maturity_time))

# spot = np.linspace(80, 120, 10)
# vol = np.linspace(.1, .3, 10)
# print(vol)
# print(spot)

call_vals = call_value(spot[np.newaxis, :], strike_price, risk_free_interest_rate, vol[:, np.newaxis], maturity_time)
put_vals = put_value(spot[np.newaxis, :], strike_price, risk_free_interest_rate, vol[:, np.newaxis], maturity_time)

df_call = round(pd.DataFrame(call_vals, index=vol, columns=spot), 2)
# print(df)

fig1, ax1 = plt.subplots()
sns.heatmap(df_call, annot=True, annot_kws={'size': 20 / np.sqrt(len(df_call))}, fmt=".2f", cmap="viridis", square=True, robust=True)
plt.title("CALL")
plt.xlabel("Spot Price")
plt.ylabel("Volatility")
plt.xticks(fontsize=6, rotation=0)
plt.yticks(fontsize=6)
plt.show()

col1, col2 = st.columns(2)
col1.header("Call Price Heatmap")
col1.pyplot(fig1)


df_put = round(pd.DataFrame(put_vals, index=vol, columns=spot), 2)
fig2, ax2 = plt.subplots()
sns.heatmap(df_put, annot=True, annot_kws={'size': 20 / np.sqrt(len(df_put))}, fmt=".2f", cmap="viridis", square=True, robust=True)
plt.title("PUT")
plt.xlabel("Spot Price")
plt.ylabel("Volatility")
plt.xticks(fontsize=6, rotation=0)
plt.yticks(fontsize=6)
plt.show()

col2.header("Call Price Heatmap")
col2.pyplot(fig2)

