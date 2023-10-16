import csv 

if __name__ == "__main__":
    with open("data/train.csv", newline="") as csvfile:
        for row in csvfile:
            print(row)