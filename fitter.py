import os
import sys
import math
import numpy as np
import pandas as pd
from matplotlib import style, pyplot as plt
import seaborn as sns
import sqlalchemy as db
from sqlalchemy.orm import Session
from sqlalchemy_utils import database_exists, create_database
from fitting import Fitting, Base
from fitting_exceptions import DataframeEmptyException, DataUnfittableException, DataframeFormatException, InvalidIndexException

class Fitter:
    """Provides functionality to fit experimental data points to a set of ideal functions and to showcase resulting data"""
    _df_train = None
    _df_ideal = None
    _df_test = None
    _count_ys_train = None
    _count_ys_ideal = None

    _indices_sum_lsq = None

    _selected_ideal_funcs = None
    _count_xs_test = None
    _count_ys_selected_ideal = None
    _df_selected_ideals = None
    _df_fittings = None
    _df_train_deltas = None

    _engine = None
    _connection = None

    def __init__(self, path_training_csv:str, path_ideal_csv:str, path_test_csv:str) -> None:
        if not (os.path.exists(path_training_csv) and os.path.exists(path_ideal_csv) and os.path.exists(path_test_csv))\
            or not (os.path.splitext(path_training_csv)[1] == ".csv" and os.path.splitext(path_ideal_csv)[1] == ".csv" and os.path.splitext(path_test_csv)[1] == ".csv"):
            raise FileNotFoundError
        
        self._load_input_(path_train=path_training_csv, path_ideal=path_ideal_csv, path_test=path_test_csv)
        self._fit_()

    def _load_input_(self, path_train:str, path_ideal:str, path_test:str):
        """Loads input data into dataframes for further processing"""
        self._df_train = pd.read_csv(path_train)
        self._df_ideal = pd.read_csv(path_ideal)
        self._df_test = pd.read_csv(path_test)
        
        if(type(self._df_train) == type(None) or type(self._df_ideal) == type(None) or type(self._df_test) == type(None)):    
            raise DataframeEmptyException("No data could be loaded for at least one dataframe")
        if not ((self._df_train.columns[0] == "x" and self._df_ideal.columns[0] == "x" and self._df_test.columns[0] == "x")):
            raise DataframeFormatException("First column for input dataframes must be labeled 'x'")
        if not (len(self._df_train.columns) >= 2 and len(self._df_ideal.columns) >= 2):
            raise DataframeFormatException("Training and ideal data needs a column for x-values and at least one other column for y-values")
        if not (self._df_train.shape[0] == self._df_ideal.shape[0]):
            raise DataframeFormatException("Training and ideal data must have the same number of x-values")
        if not (self._df_train.shape[0] == self._df_train["x"].nunique()):
            raise InvalidIndexException("X-values for training and ideal data must be unique")

        self._count_ys_train = self._df_train.shape[1] - 1
        self._count_ys_ideal = self._df_ideal.shape[1] - 1

    def _sum_of_least_squares_(self):
        """Checks which ideal function best fits which training function"""
        sum = np.empty(shape=(self._count_ys_train, self._count_ys_ideal))
        sum_lsq = np.full(shape=self._count_ys_train, fill_value=sys.float_info.max)
        self._indices_sum_lsq = np.full_like(sum_lsq, -1, dtype="int16")
        for y_train in range(0, self._count_ys_train):
            for col_ideal in range(0, self._count_ys_ideal):
                sum[y_train, col_ideal] = np.sum((self._df_train.iloc[:, y_train + 1] \
                                                - self._df_ideal.iloc[:, col_ideal + 1]) ** 2)
                if sum[y_train, col_ideal] < sum_lsq[y_train]:
                    sum_lsq[y_train] = sum[y_train, col_ideal]
                    self._indices_sum_lsq[y_train] = col_ideal

    def _validate_selection_(self):
        """Check which test data point fits which selected ideal function"""
        self._selected_ideal_funcs = self._df_ideal.iloc[:, self._indices_sum_lsq]
        self._count_xs_test = self._df_test.shape[0]
        self._count_ys_selected_ideal = self._selected_ideal_funcs.shape[1]
        # adding x-column next to the y-columns of selected ideal functions to enable indexing by x_test value
        self._df_selected_ideals = pd.DataFrame(data=np.hstack((np.atleast_2d(self._df_ideal["x"]).T, self._selected_ideal_funcs)), \
                                            columns=np.hstack(("x", self._selected_ideal_funcs.columns)), dtype="float32")
        # initialize deltas as float max to be able to tell easily if values aren't getting filled in
        self._df_test_deltas = pd.DataFrame(data=np.full(shape=(self._count_xs_test, self._count_ys_selected_ideal), fill_value=sys.float_info.max), \
                                        index=self._df_test["x"], columns=self._df_selected_ideals.columns[1:], dtype="float32")
        # calculate respective deltas between training function and its selected ideal function
        train_deltas = np.full(shape=self._df_train.iloc[:, 1:].shape, fill_value=sys.float_info.max)
        
        for col_index in range(self._count_ys_selected_ideal):
            train_deltas[:, col_index] = (self._df_train.iloc[:, col_index + 1] - self._selected_ideal_funcs.iloc[:, col_index]) ** 2
        self._df_train_deltas = pd.DataFrame(data=train_deltas, index=self._df_train["x"], columns=self._selected_ideal_funcs.columns, dtype="float32")
        
        # container for final table data
        self._df_fittings = pd.DataFrame(np.empty(shape=(self._count_xs_test, 4)), columns=["x", "y", "delta", "ideal_func"], dtype="float32")
        self._df_fittings = self._df_fittings.astype({"ideal_func": "string"})
        # calculating residuals of test data compared to selected ideal functions
        for test_index in range(self._count_xs_test):
            x_test = self._df_test.iloc[test_index]["x"]
            # get delta of selected ideal y-values at this particular x-value and respective test y-value
            self._df_test_deltas.iloc[test_index] = (self._df_selected_ideals[self._df_selected_ideals["x"] == x_test].iloc[:, 1:] \
                                                - self._df_test.iloc[test_index]["y"]) ** 2
            # take note of which ideals this test data point can be fitted to
            col_indices_fittable = np.where((self._df_train_deltas.loc[x_test] * math.sqrt(2)) \
                                            - self._df_test_deltas.iloc[test_index] > 0)
            # take note of fittable ideal with the closest fit
            col_index_fittable_min_delta = np.where(self._df_test_deltas.iloc[test_index] \
                                                    == np.min(self._df_test_deltas.iloc[test_index, col_indices_fittable[0]]))
            self._df_fittings.iloc[test_index, 0] = self._df_test["x"].iloc[test_index].astype("float32")
            self._df_fittings.iloc[test_index, 1] = self._df_test["y"].iloc[test_index].astype("float32")
            if(len(col_index_fittable_min_delta[0]) > 0):
                self._df_fittings.iloc[test_index, 2] = self._df_test_deltas.iloc[test_index, col_index_fittable_min_delta[0][0]].astype("float32")
                self._df_fittings.iloc[test_index, 3] = self._df_test_deltas.columns[col_index_fittable_min_delta[0][0]]
            else:
                col_index_min_delta = np.where(self._df_test_deltas.iloc[test_index] \
                                                == np.min(self._df_test_deltas.iloc[test_index]))
                self._df_fittings.iloc[test_index, 2] = self._df_test_deltas.iloc[test_index, col_index_min_delta[0][0]].astype("float32")
                self._df_fittings.iloc[test_index, 3] = "N/A"

    def visualize(self):
        """Displays graphs showcasing test data points with their most fittable ideal functions and those ideal functions compared to the training functions used to select them"""
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(nrows=2, ncols=self._count_ys_selected_ideal, sharey="none", sharex="all")
        # row 1: test data points w/ their selected ideal functions 
        for sel_col_index in range(self._count_ys_selected_ideal):
            associated_points = self._df_fittings[self._df_fittings["ideal_func"] == self._selected_ideal_funcs.columns[sel_col_index]]
            for point_index in range(associated_points.shape[0]):
            # draw vertical lines for each x_test between y_test and y_ideal
                x_coord = associated_points["x"].iloc[point_index]
                x_coords = [x_coord, x_coord]
                y_coords = [associated_points["y"].iloc[point_index], \
                        self._df_selected_ideals[self._df_selected_ideals["x"] == associated_points["x"].iloc[point_index]][associated_points.iloc[point_index]["ideal_func"]].iloc[0]]
                # add the slightest horizontal offset to one end of the line since true verticals aren't rendered properly
                x_coords[0] = x_coords[0] - x_coords[0]/10000         
                sns.lineplot(x=x_coords, y=y_coords, ax=ax[0, sel_col_index], linewidth=0.5, linestyle=":", color="#ff6969")
            sns.scatterplot(data=associated_points, x=associated_points["x"], y=associated_points["y"], hue=associated_points["delta"], ax=ax[0, sel_col_index], size=1, legend=False)
            sns.lineplot(x=self._df_selected_ideals["x"], y=self._df_selected_ideals.iloc[:, sel_col_index + 1], ax=ax[0, sel_col_index], linewidth=0.5, linestyle="-")\
                .set(title=self._df_selected_ideals.columns[sel_col_index + 1], ylabel="")
            ax[0,0].set(ylabel="")
        ax[0,0].set(ylabel="y")

        # row 2: training functions & selected ideal functions
        for col_index in range(self._count_ys_selected_ideal):
            if(col_index == self._count_ys_selected_ideal - 1):
                sns.lineplot(x=self._df_selected_ideals["x"], y=self._df_selected_ideals.iloc[:, col_index + 1], ax=ax[1, col_index], linewidth=0.5, linestyle="-", label="ideal")
                sns.lineplot(x=self._df_train["x"], y=self._df_train.iloc[:, col_index + 1], ax=ax[1, col_index], linewidth=1, linestyle="-", label="train")
            else:
                sns.lineplot(x=self._df_selected_ideals["x"], y=self._df_selected_ideals.iloc[:, col_index + 1], ax=ax[1, col_index], linewidth=0.5, linestyle="-")
                sns.lineplot(x=self._df_train["x"], y=self._df_train.iloc[:, col_index + 1], ax=ax[1, col_index], linewidth=1, linestyle="-")
            ax[1, col_index].set(ylabel="")
        ax[1, 0].set(ylabel="y")

        fig.suptitle("Selected ideal functions")
        plt.legend()
        plt.show()

    def export_fittings_to_db(self, connection_string:str=None):
        self._engine = db.create_engine(connection_string, echo=False)
        if not database_exists(self._engine.url):
            create_database(self._engine.url)
        self._connection = self._engine.connect()
        Base.metadata.create_all(bind=self._engine)
        with Session(bind=self._engine) as session:
            if type(self._df_fittings) == type(None):
                raise DataframeEmptyException("An error occured during calculation of fittings dataframe")
            for index in range(self._count_xs_test):
                # If entry at index already exists, update it
                if(session.query(db.exists().where(Fitting.id == index)).scalar()):
                    session.execute(db.update(Fitting).where(Fitting.id == index).values(x = str(self._df_fittings.iloc[index]["x"]),
                                                                                         y = str(self._df_fittings.iloc[index]["y"]),
                                                                                         delta = str(self._df_fittings.iloc[index]["delta"]),
                                                                                         ideal_function = self._df_fittings.iloc[index]["ideal_func"]))
                # If entry does not yet exist, create it
                else:
                    fitting_entry = Fitting(
                        id = index,
                        x = str(self._df_fittings.iloc[index]["x"]),
                        y = str(self._df_fittings.iloc[index]["y"]),
                        delta = str(self._df_fittings.iloc[index]["delta"]),
                        ideal_function = self._df_fittings.iloc[index]["ideal_func"]
                    )
                    session.add(fitting_entry)
            session.commit()

    def _fit_(self):
        self._sum_of_least_squares_()
        self._validate_selection_()