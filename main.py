from fitter import Fitter

path_train = "data/train.csv"
path_ideal = "data/ideal.csv"
path_test = "data/test.csv"

if __name__ == "__main__":
    fitter = Fitter(path_train, path_ideal, path_test)
    if(fitter.export_fittings_to_db("sqlite:///fittings.sqlite") == False):
        exit()
    fitter.visualize()