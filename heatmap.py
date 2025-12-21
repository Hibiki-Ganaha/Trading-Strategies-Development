import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("AAPL History")

data = np.random.random((12,12))
print(data)

fig, ax = plt.subplots()
