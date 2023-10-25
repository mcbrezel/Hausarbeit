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
    selected_ideal_funcs = df_ideal.iloc[:, sum_lsq_indices]
    count_xs_test = df_test.shape[0]
    count_ys_sel_ideal = selected_ideal_funcs.shape[1]
    test_deltas = np.full(shape=(count_xs_test, count_ys_sel_ideal), fill_value=sys.float_info.max)
    train_deltas = np.full(shape=df_train.iloc[:, 1:].shape, fill_value=sys.float_info.max)
    max_test_deltas = np.full(shape=count_ys_sel_ideal, fill_value=sys.float_info.max)
    max_train_deltas = np.full_like(a=max_test_deltas, fill_value=sys.float_info.max)
    
    for col_sel_ideal in range(count_ys_sel_ideal):
        for row_test in range(count_xs_test):
            x_test, y_test = df_test.iloc[row_test, :] 
            test_deltas[row_test, col_sel_ideal] = (selected_ideal_funcs[df_ideal["x"] == x_test].iloc[:, col_sel_ideal] - y_test) ** 2
        max_test_deltas[col_sel_ideal] = np.max(test_deltas[:, col_sel_ideal])
        train_deltas[:, col_sel_ideal] = (df_train.iloc[:, col_sel_ideal + 1] - selected_ideal_funcs.iloc[:, col_sel_ideal]) ** 2
        max_train_deltas[col_sel_ideal] = np.max(train_deltas[:, col_sel_ideal])

    # adding x-column next to the y-columns of selected ideal functions
    df_sel_ideal = pd.DataFrame(np.hstack((np.atleast_2d(df_ideal["x"]).T, selected_ideal_funcs)), columns=np.hstack(("x", selected_ideal_funcs.columns)))
    residuals = pd.DataFrame(data=np.full(shape=(count_xs_test, count_ys_sel_ideal), fill_value=-1), index=df_test["x"], columns=df_sel_ideal.columns[1:], dtype="float32")
    # calculating residuals of test data compared to selected ideal functions
    for x_test in df_test["x"]:
        for index in np.where(df_test["x"] == x_test)[0]:
            residuals.iloc[index] = (df_sel_ideal[df_sel_ideal["x"] == x_test].iloc[:, 1:] - df_test.loc[df_test["x"] == x_test]["y"][index]) ** 2
    print(residuals)