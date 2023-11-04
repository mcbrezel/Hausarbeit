from base_fitting import Base_Fitting

class Fitting(Base_Fitting):
    def __init__(self, x: float, y: float, delta: float, ideal_function: str) -> None:
        super().__init__(x, y, delta, ideal_function)