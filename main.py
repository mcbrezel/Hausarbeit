import csv
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from tables import Base, TrainingData
import pandas as pd
from matplotlib import style, pyplot as plot

path_train = "data/train.csv"
default_chunksize = 100

if __name__ == "__main__":
    dataframe_train = pd.read_csv(path_train)
    
    style.use(style="ggplot")
    fig, ax = plot.subplots(3)
    for i in range(1, dataframe_train.shape[1]):
        ax[0].plot(dataframe_train.iloc[:,0], dataframe_train.iloc[:,i], \
                     label="y" + str(i), linewidth=2)
    ax[0].legend()
    ax[0].grid(True, color="k")
    plot.ylabel("y axis")
    plot.xlabel("x axis")
    plot.title("Training data")
    plot.show()