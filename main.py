import csv
import numpy as np
import pandas as pd
from matplotlib import style, pyplot as plot
import seaborn as sb

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

    width_train = dataframe_train.shape[1]
    width_ideal = dataframe_ideal.shape[1]
    sum = np.empty(shape=(width_train - 1, width_ideal - 1))
    for y_train in range(1, dataframe_train.shape[1]):
        for y_ideal in range(1, dataframe_ideal.shape[1]):
            sum[y_train - 1, y_ideal - 1] = np.sum((dataframe_train.iloc[:,y_train] - dataframe_ideal.iloc[:,y_ideal])**2)

    for item in sum:
        print(item)