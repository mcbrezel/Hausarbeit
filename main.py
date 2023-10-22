import csv
import numpy as np
import pandas as pd
from matplotlib import style, pyplot as plot
import seaborn as sb
import sys

path_train = "data/train.csv"
path_ideal = "data/ideal.csv"
path_test = "data/test.csv"

if __name__ == "__main__":
    dataframe_train = pd.read_csv(path_train)
    dataframe_ideal = pd.read_csv(path_ideal)
    dataframe_test = pd.read_csv(path_test)

    # style.use(style="ggplot")
    # train_melted = dataframe_train.melt(id_vars="x", var_name="functions", value_name="y")
    # ideal_melted = dataframe_ideal.melt(id_vars="x", var_name="functions", value_name="y")
    # sb.relplot(data=train_melted, x="x", y="y", hue="functions", kind="line")
    # sb.relplot(data=ideal_melted, x="x", y="y", hue="functions", kind="line")
    # sb.relplot(data=dataframe_test, x="x", y="y")
    # plot.show()

    # 1) Sum of Least Squares (SLS)
    # Check which ideal function best fits which training function
    count_ys_train = dataframe_train.shape[1] - 1
    count_ys_ideal = dataframe_ideal.shape[1] - 1
    sum = np.empty(shape=(count_ys_train, count_ys_ideal))
    sls = np.full(shape=count_ys_train, fill_value=sys.float_info.max)
    sls_index = np.full_like(sls, -1)
    for y_train in range(0, count_ys_train):
        for y_ideal in range(0, count_ys_ideal):
            sum[y_train, y_ideal] = np.sum((dataframe_train.iloc[:, y_train + 1] - dataframe_ideal.iloc[:, y_ideal + 1]) ** 2)
            if sum[y_train, y_ideal] < sls[y_train]:
                sls[y_train] = sum[y_train, y_ideal]
                sls_index[y_train] = y_ideal 