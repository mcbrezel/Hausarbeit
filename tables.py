from typing import Any
from sqlalchemy import Column, Float
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column

class Base(DeclarativeBase):
    pass        

class TrainingData(Base):
    __tablename__ = "training_data"
    x: Mapped[Float] = mapped_column("x", Float, primary_key=True)
    y1: Mapped[Float] = mapped_column("y1", Float)
    y2: Mapped[Float] = mapped_column("y2", Float)
    y3: Mapped[Float] = mapped_column("y3", Float)
    y4: Mapped[Float] = mapped_column("y4", Float)