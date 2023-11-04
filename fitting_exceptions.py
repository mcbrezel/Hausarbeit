

class BaseDataframeException(Exception):
    """Common baseclass for exceptions related to dataframe errors"""
    message = ""

    def __init__(self, message:str=None, *args: object) -> None:
        super().__init__(*args)
        self.message = message

class DataUnfittableException(BaseDataframeException):
    """Maximum delta for test values exceeds fittability threshold"""
    pass

class DataframeEmptyException(BaseDataframeException):
    """Dataframe does not contain any data"""
    pass

class DataframeFormatException(BaseDataframeException):
    """Dataframe has incorrect shape or format"""
    pass

class InvalidIndexException(BaseDataframeException):
    """Index column values must be unique"""
    pass