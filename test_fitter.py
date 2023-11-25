import unittest
from fitter import Fitter
from fitting_exceptions import DataframeEmptyException

class Test_Fitter(unittest.TestCase):
    path_train = "data/train.csv"
    path_ideal = "data/ideal.csv"
    path_test = "data/test.csv"

    def test_CheckInputFileValidity_WithTrainWrongFileExtension_RaisesFileNotFoundError(self):
        with self.assertRaises(FileNotFoundError):
            f = Fitter("data/train.txt", self.path_ideal, self.path_test)
    
    def test_CheckInputFileValidity_WithIdealWrongFileExtension_RaisesFileNotFoundError(self):
        with self.assertRaises(FileNotFoundError):
            f = Fitter(self.path_train, "data/ideal.txt", self.path_test)
    
    def test_CheckInputFileValidity_WithTestWrongFileExtension_RaisesFileNotFoundError(self):
        with self.assertRaises(FileNotFoundError):
            f = Fitter(self.path_train, self.path_ideal, "data/test.txt")

    def test_CheckInputFileValidity_WithNonexistentTrainFileName_RaisesFileNotFoundError(self):
        with self.assertRaises(FileNotFoundError):
            f = Fitter("data/nonexistent.csv", self.path_ideal, self.path_test)

    def test_CheckInputFileValidity_WithNonexistentIdealFileName_RaisesFileNotFoundError(self):
        with self.assertRaises(FileNotFoundError):
            f = Fitter(self.path_train, "data/nonexistent.csv", self.path_test)

    def test_CheckInputFileValidity_WithNonexistentTestFileName_RaisesFileNotFoundError(self):
        with self.assertRaises(FileNotFoundError):
            f = Fitter(self.path_train, self.path_ideal, "data/nonexistent.csv")

    def test_CheckInputFileValidity_WithValidFileNames_ReturnsTrue(self):
        f = Fitter(self.path_train, self.path_ideal, self.path_test)
        self.assertTrue(f._check_input_file_validity_(self.path_train, self.path_ideal, self.path_test))

    def test_ExportFittingsToDb_WithDfFittingsMissing_RaisesDataframeEmptyException(self):
        # Arrange
        f = Fitter(self.path_train, self.path_ideal, self.path_test)
        f._df_fittings = None
        # Assert
        with self.assertRaises(DataframeEmptyException):
            # Act
            f.export_fittings_to_db("sqlite:///fittings.sqlite")


if __name__ == "__main__":
    unittest.main()