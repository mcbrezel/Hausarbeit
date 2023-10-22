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
    df_train = pd.read_csv(path_train)
    df_ideal = pd.read_csv(path_ideal)
    df_test = pd.read_csv(path_test)

    # style.use(style="ggplot")
    # train_melted = dataframe_train.melt(id_vars="x", var_name="functions", value_name="y")
    # ideal_melted = dataframe_ideal.melt(id_vars="x", var_name="functions", value_name="y")
    # sb.relplot(data=train_melted, x="x", y="y", hue="functions", kind="line")
    # sb.relplot(data=ideal_melted, x="x", y="y", hue="functions", kind="line")
    # sb.relplot(data=dataframe_test, x="x", y="y")
    # plot.show()

    # 1) Sum of Least Squares (SLS)
    # Check which ideal function best fits which training function
    count_ys_train = df_train.shape[1] - 1
    count_ys_ideal = df_ideal.shape[1] - 1
    sum = np.empty(shape=(count_ys_train, count_ys_ideal))
    sum_lsq = np.full(shape=count_ys_train, fill_value=sys.float_info.max)
    sum_lsq_indices = np.full_like(sum_lsq, -1, dtype="int16")
    for y_train in range(0, count_ys_train):
        for col_ideal in range(0, count_ys_ideal):
            sum[y_train, col_ideal] = np.sum((df_train.iloc[:, y_train + 1] - df_ideal.iloc[:, col_ideal + 1]) ** 2)
            if sum[y_train, col_ideal] < sum_lsq[y_train]:
                sum_lsq[y_train] = sum[y_train, col_ideal]
                sum_lsq_indices[y_train] = col_ideal 
    
    # 2) Selection validation
    # Check which test data point fits which selected ideal function
    count_xs_test = df_test.shape[0]
    selected_ideal_funcs = df_ideal.iloc[:, sum_lsq_indices]
    test_deltas = np.full(shape=(count_xs_test, selected_ideal_funcs.shape[1]), fill_value=sys.float_info.max)
    max_deltas = np.full(shape=selected_ideal_funcs.shape, fill_value=sys.float_info.max)
    for col_ideal in range(selected_ideal_funcs.shape[1]):
        for row_test in range(count_xs_test):
            x_test, y_test = df_test.iloc[row_test, :] 
            test_deltas[row_test, col_ideal] = (selected_ideal_funcs[df_ideal["x"] == x_test].iloc[:, col_ideal] - y_test) ** 2
        max_deltas[col_ideal] = np.max(test_deltas[col_ideal])
    