import csv
from itertools import islice

with open("./train.csv") as f_train:
    with open("./data/test.csv", "a") as f_test:
        data = csv.writer(f_test)
        for i, row in enumerate(islice(csv.reader(f_train), 10000, 12000)):
            row_w = [i]+ row
            data.writerow(row_w)

with open("./train.csv") as f_train:
    with open("./data/train1.csv", "a") as f_test:
        data = csv.writer(f_test)
        for row in islice(csv.reader(f_train), 0, 10000):
            data.writerow(row)