from sqlalchemy import String, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class Fitting(Base):
    __tablename__ = "Fitting"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    x: Mapped[str] = mapped_column(String)
    y: Mapped[str] = mapped_column(String)
    delta: Mapped[str] = mapped_column(String)
    ideal_function: Mapped[str] = mapped_column(String)

    def __init__(self, id:int, x:str, y:str, delta:str, ideal_function:str) -> None:
        self.id = id
        self.x = x
        self.y = y
        self.delta = delta
        self.ideal_function = ideal_function

    def __repr__(self) -> str:
        """Returns instance data in a human-readable form"""
        return "x: {0}, y: {1}, delta: {2}, ideal function: {3}".format(self.x, self.y, self.delta, self.ideal_function)