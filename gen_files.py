"""
This file is a helper for pre-processing the dataset.
"""

import csv
from itertools import islice


# Extract all images with emotions of angry, sad, happy or neutral
with open("./train.csv") as f_train:
    with open("./train_small.csv", "a") as f_test:
        data = csv.writer(f_test)
        for row in islice(csv.reader(f_train), 1, 28710):
            row_copy = [i for i in row]

            if row_copy[0] == '0':    
                data.writerow(row_copy)
            elif row_copy[0] == '3':
                row_copy[0] = 1
                data.writerow(row_copy)
            elif row_copy[0] == '4':
                row_copy[0] = 2
                data.writerow(row_copy)
            elif row_copy[0] == '6':
                row_copy[0] = 3
                data.writerow(row_copy)

# Generate test set
with open("./train_small.csv") as f_train:
    with open("./data/test.csv", "a") as f_test:
        data = csv.writer(f_test)
        for i, row in enumerate(islice(csv.reader(f_train), 20000, 21000)):
            row_w = [i]+ row
            data.writerow(row_w)

# Generate training set
with open("./train_small.csv") as f_train:
    with open("./data/train1.csv", "a") as f_test:
        data = csv.writer(f_test)
        for row in islice(csv.reader(f_train), 0, 20000):
            data.writerow(row)