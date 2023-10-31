import csv
import numpy as np
import pandas as pd
from matplotlib import style, pyplot as plot
import seaborn as sns
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
    # sns.relplot(data=train_melted, x="x", y="y", hue="functions", kind="line")
    # sns.relplot(data=ideal_melted, x="x", y="y", hue="functions", kind="line")
    # sns.relplot(data=dataframe_test, x="x", y="y")
    # plot.show()

    # 1) Sum of Least Squares (SLS)
    # Check which ideal function best fits which training function
    ################################################################
    count_ys_train = df_train.shape[1] - 1
    count_ys_ideal = df_ideal.shape[1] - 1
    sum = np.empty(shape=(count_ys_train, count_ys_ideal))
    sum_lsq = np.full(shape=count_ys_train, fill_value=sys.float_info.max)
    indices_sum_lsq = np.full_like(sum_lsq, -1, dtype="int16")
    for y_train in range(0, count_ys_train):
        for col_ideal in range(0, count_ys_ideal):
            sum[y_train, col_ideal] = np.sum((df_train.iloc[:, y_train + 1] \
                                              - df_ideal.iloc[:, col_ideal + 1]) ** 2)
            if sum[y_train, col_ideal] < sum_lsq[y_train]:
                sum_lsq[y_train] = sum[y_train, col_ideal]
                indices_sum_lsq[y_train] = col_ideal 
    
    # 2) Selection validation
    # Check which test data point fits which selected ideal function
    ################################################################
    selected_ideal_funcs = df_ideal.iloc[:, indices_sum_lsq]
    count_xs_test = df_test.shape[0]
    count_ys_selected_ideal = selected_ideal_funcs.shape[1]
    # adding x-column next to the y-columns of selected ideal functions to enable indexing by x_test value
    df_selected_ideals = pd.DataFrame(data=np.hstack((np.atleast_2d(df_ideal["x"]).T, selected_ideal_funcs)), \
                                      columns=np.hstack(("x", selected_ideal_funcs.columns)), dtype="float32")
    # initialize deltas as float max to be able to tell easily if values aren't getting filled in
    df_test_deltas = pd.DataFrame(data=np.full(shape=(count_xs_test, count_ys_selected_ideal), fill_value=sys.float_info.max), \
                                  index=df_test["x"], columns=df_selected_ideals.columns[1:], dtype="float32")
    max_test_deltas = np.full(shape=count_ys_selected_ideal, fill_value=sys.float_info.max, dtype="float32")
    # calculate respective deltas between training function and its selected ideal function
    train_deltas = np.full(shape=df_train.iloc[:, 1:].shape, fill_value=sys.float_info.max)
    max_train_deltas = np.full_like(a=max_test_deltas, fill_value=sys.float_info.max)
    for col_index in range(count_ys_selected_ideal):
        train_deltas[:, col_index] = (df_train.iloc[:, col_index + 1] - selected_ideal_funcs.iloc[:, col_index]) ** 2
        max_train_deltas[col_index] = np.max(train_deltas[:, col_index])
    df_train_deltas = pd.DataFrame(data=train_deltas, index=df_train["x"], \
                                   columns=selected_ideal_funcs.columns, dtype="float32")
    
    # list of smallest fittable deltas for each test data point for when there are multiple fits
    df_fittable_min_delta = pd.DataFrame(data=np.empty(shape=(count_xs_test, 2)))
    # container for final table data
    df_fittings = pd.DataFrame(np.empty(shape=(count_xs_test, 4)), columns=["x", "y", "delta", "ideal_func"], dtype="float32")
    df_fittings = df_fittings.astype({"ideal_func": "string"})
    # calculating residuals of test data compared to selected ideal functions
    for x_coord in df_test["x"]:
        for test_index in np.where(df_test["x"] == x_coord)[0]:
            df_test_deltas.iloc[test_index] = (df_selected_ideals[df_selected_ideals["x"] == x_coord].iloc[:, 1:] \
                                               - df_test.loc[df_test["x"] == x_coord]["y"][test_index]) ** 2
            col_indices_fittable = np.where((df_train_deltas.loc[x_coord] * math.sqrt(2)) \
                                            - df_test_deltas.loc[x_coord] > 0)
            df_fittable_deltas = df_test_deltas.iloc[test_index].iloc[col_indices_fittable[0]]
            col_index_fittable_min_delta = np.where(df_test_deltas.iloc[test_index] \
                                                    == np.min(df_test_deltas.iloc[test_index]))
            df_fittings.iloc[test_index, 0] = df_test["x"].iloc[test_index].astype("float32")
            df_fittings.iloc[test_index, 1] = df_test["y"].iloc[test_index].astype("float32")
            df_fittings.iloc[test_index, 2] = df_test_deltas.iloc[test_index, col_index_fittable_min_delta[0][0]].astype("float32")
            df_fittings.iloc[test_index, 3] = df_test_deltas.columns[col_index_fittable_min_delta[0][0]]

    # Visualization
    ################################################################
    fig, ax = plot.subplots(nrows=2, ncols=count_ys_selected_ideal, sharey="row", sharex="row")
    # row 1: test data points w/ their selected ideal functions 
    for index_sel_col in range(count_ys_selected_ideal):
        associated_points = df_fittings[df_fittings["ideal_func"] == selected_ideal_funcs.columns[index_sel_col]]
        for point_index in range(associated_points.shape[0]):
        # draw vertical lines for each x_test between y_test and y_ideal
            x_coord = associated_points["x"].iloc[point_index]
            x_coords = [x_coord, x_coord]
            #           y_test                          , y_ideal
            y_coords = [associated_points["y"].iloc[point_index], \
                       df_selected_ideals[df_selected_ideals["x"] == associated_points["x"].iloc[point_index]][associated_points.iloc[point_index]["ideal_func"]].iloc[0]]
            # add the slightest horizontal offset to one end of the line since true verticals aren't rendered properly
            x_coords[0] = x_coords[0] - x_coords[0]/1000         
            sns.lineplot(x=x_coords, y=y_coords, ax=ax[0, index_sel_col], linewidth=0.5, linestyle=":")
        sns.scatterplot(data=associated_points, x=associated_points["x"], y=associated_points["y"], ax=ax[0, index_sel_col], size=1, legend=False)
        sns.lineplot(x=df_selected_ideals["x"], y=df_selected_ideals.iloc[:, index_sel_col + 1], ax=ax[0, index_sel_col], linewidth=1, linestyle="-")
    
    # row 2: training functions & selected ideal functions
    for col_index in range(count_ys_selected_ideal):
        sns.lineplot(x=df_selected_ideals["x"], y=df_selected_ideals.iloc[:, col_index + 1], ax=ax[1, col_index], linewidth=0.5, linestyle=":")
    for col_index in range(count_ys_selected_ideal):
        sns.lineplot(x=df_train["x"], y=df_train.iloc[:, col_index + 1], ax=ax[1, col_index], linewidth=1, linestyle="-")


    plot.show()