import csv
import numpy as np
import pandas as pd
from matplotlib import style, pyplot as plot
import seaborn as sb
import sys
import math
from fitting import Fitting

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
    ##############################################################
    count_ys_train = df_train.shape[1] - 1
    count_ys_ideal = df_ideal.shape[1] - 1
    sum = np.empty(shape=(count_ys_train, count_ys_ideal))
    sum_lsq = np.full(shape=count_ys_train, fill_value=sys.float_info.max)
    sum_lsq_indices = np.full_like(sum_lsq, -1, dtype="int16")
    for y_train in range(0, count_ys_train):
        for col_ideal in range(0, count_ys_ideal):
            sum[y_train, col_ideal] = np.sum((df_train.iloc[:, y_train + 1] \
                                              - df_ideal.iloc[:, col_ideal + 1]) ** 2)
            if sum[y_train, col_ideal] < sum_lsq[y_train]:
                sum_lsq[y_train] = sum[y_train, col_ideal]
                sum_lsq_indices[y_train] = col_ideal 
    
    # 2) Selection validation
    # Check which test data point fits which selected ideal function
    ################################################################
    selected_ideal_funcs = df_ideal.iloc[:, sum_lsq_indices]
    count_xs_test = df_test.shape[0]
    count_ys_sel_ideal = selected_ideal_funcs.shape[1]
    # adding x-column next to the y-columns of selected ideal functions to enable indexing by x_test value
    df_sel_ideal = pd.DataFrame(data=np.hstack((np.atleast_2d(df_ideal["x"]).T, selected_ideal_funcs)), \
                                columns=np.hstack(("x", selected_ideal_funcs.columns)))
    train_deltas = np.full(shape=df_train.iloc[:, 1:].shape, \
                           fill_value=sys.float_info.max)
    df_train_deltas = pd.DataFrame(data=train_deltas, \
                                   index=df_train["x"], \
                                   columns=selected_ideal_funcs.columns, \
                                   dtype="float32")
    df_test_deltas = pd.DataFrame(data=np.full(shape=(count_xs_test, count_ys_sel_ideal), fill_value=-1), \
                                  index=df_test["x"], \
                                  columns=df_sel_ideal.columns[1:], \
                                  dtype="float32")
    # list of smallest fittable deltas for each test data point for when there are multiple fits
    df_fittable_min_delta = pd.DataFrame(data=np.empty(shape=(count_xs_test, 2)))
    # container for final table data
    fittings = []
    # calculating residuals of test data compared to selected ideal functions
    for x_test in df_test["x"]:
        for test_index in np.where(df_test["x"] == x_test)[0]:
            df_test_deltas.iloc[test_index] = (df_sel_ideal[df_sel_ideal["x"] == x_test].iloc[:, 1:] \
                                               - df_test.loc[df_test["x"] == x_test]["y"][test_index]) ** 2            
            col_indices_fittable = np.where((df_train_deltas.loc[x_test] * math.sqrt(2)) \
                                            - df_test_deltas.loc[x_test] > 0)
            df_fittable_deltas = df_test_deltas.iloc[test_index].iloc[col_indices_fittable[0]]
            col_index_fittable_min_delta = np.where(df_test_deltas.iloc[test_index] \
                                                    == np.min(df_test_deltas.iloc[test_index]))
            fittings.append(Fitting(x=x_test, \
                                    y=df_test["y"].iloc[test_index], \
                                    delta=df_test_deltas.iloc[test_index, col_index_fittable_min_delta[0][0]], \
                                    ideal_function=df_test_deltas.columns[col_index_fittable_min_delta[0][0]]))

    for f in fittings:
        print(f)