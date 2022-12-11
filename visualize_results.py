"""This file is used to visualize the result."""
import matplotlib.pyplot as plt

classes = ["Angry", "Happy", "Sad", "Neutral"]
results = []

with open("results.txt") as r:
    results = r.read().splitlines()

# Visualize 5 images.
for i in range(5):
    img = plt.imread(f"./data/test/test{i}.jpg")
    plt.title(f"Prediction: {classes[int(results[i])]}")
    plt.imshow(img, cmap="gray")
    plt.show()