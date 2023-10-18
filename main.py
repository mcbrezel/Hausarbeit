import csv
from numpy import genfromtxt
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from tables import Base, TrainingData
import pandas as pd

path_train = "data/train.csv"

def load_data(path: str) -> list:
   data = genfromtxt(path, delimiter=",", skip_header=1, converters={0: lambda s: str(s)})
   return data.tolist()

if __name__ == "__main__":
    engine = create_engine("sqlite:///?DataSource=./data/database.db", echo=True)
    Base.metadata.create_all(engine)
    
    dataframe = pd.read_csv(path_train)
    dataframe.to_sql(name=TrainingData.__tablename__, con=engine, if_exists="replace", index= False, chunksize=25, method="multi")