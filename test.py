import time

start_time = time.time()
import pandas as pd
print(f"pandas imported in {time.time() - start_time:.2f} seconds")

start_time = time.time()
import numpy as np
print(f"numpy imported in {time.time() - start_time:.2f} seconds")

start_time = time.time()
from sklearn.linear_model import LinearRegression
print(f"LinearRegression imported in {time.time() - start_time:.2f} seconds")

start_time = time.time()
from sklearn.metrics import mean_squared_error, f1_score, r2_score, roc_auc_score
print(f"sklearn.metrics imported in {time.time() - start_time:.2f} seconds")

start_time = time.time()
import catboost as cb
print(f"catboost imported in {time.time() - start_time:.2f} seconds")

start_time = time.time()
import matplotlib.pyplot as plt
print(f"matplotlib imported in {time.time() - start_time:.2f} seconds")
