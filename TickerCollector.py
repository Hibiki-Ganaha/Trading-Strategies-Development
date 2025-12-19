import yfinance as yf
import pandas as pd
import plotly.express as px

data = yf.Tickers(["AA","CAT","DIS","GM","HPQ","JNJ","MCD","MMM","MO","MRK","MSFT","PFE","PG","T","XOM"])
dow_jones_history_1yr = round(data.history("1y")["Close"],2)
print(dow_jones_history_1yr)

dow_jones_history_1yr.to_csv("dowPortfolio.csv")

fig1 = px.line(dow_jones_history_1yr)
fig1.show()