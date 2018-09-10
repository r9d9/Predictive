import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def save_errors(test,predicted):
    rates = pd.DataFrame(np.zeros((1,5)),columns =["MSE","RMSE","R2","RMSE % of mean","Cali"])
    rates["MSE"] = mean_squared_error(test, predicted)
    rates["RMSE"] = np.sqrt(((test - predicted) ** 2).mean())
    rates["R2"] = r2_score(test, predicted)
    rates["RMSE % of mean"] = np.sqrt(((test - predicted) ** 2).mean()) / test.mean()
    rates["Cali"] = predicted.mean() / test.mean()
    return rates


def print_errors(test, predicted):
    print("MSE: ", mean_squared_error(test, predicted))
    print("RMSE: ", np.sqrt(((test - predicted) ** 2).mean()))
    print("R2: ", r2_score(test, predicted))
    print("RMSE % of mean:", np.sqrt(((test - predicted) ** 2).mean()) / test.mean())
    print("Calibration:", predicted.mean() / test.mean())
