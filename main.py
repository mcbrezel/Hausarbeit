import csv
import numpy as np
import pandas as pd
from matplotlib import style, pyplot as plot
import seaborn as sb

path_train = "data/train.csv"
path_ideal = "data/ideal.csv"
path_test = "data/test.csv"
default_chunksize = 100

if __name__ == "__main__":
    dataframe_train = pd.read_csv(path_train)
    dataframe_ideal = pd.read_csv(path_ideal)
    dataframe_test = pd.read_csv(path_test)

    style.use(style="ggplot")
    train_melted = dataframe_train.melt(id_vars="x", var_name="functions", value_name="y")
    ideal_melted = dataframe_ideal.melt(id_vars="x", var_name="functions", value_name="y")
    sb.relplot(data=train_melted, x="x", y="y", hue="functions", kind="line")
    sb.relplot(data=ideal_melted, x="x", y="y", hue="functions", kind="line")
    sb.relplot(data=dataframe_test, x="x", y="y")
    plot.show()