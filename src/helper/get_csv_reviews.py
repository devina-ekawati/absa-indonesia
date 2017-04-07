import csv

reviews = []

file = "../../data/category_extraction/train_data.csv"
with open (file, "rb") as f:
	reader = csv.reader(f, delimiter=';', quotechar='"')
	next(reader)
	for row in reader:
		reviews.append(row[0])

with open("../../data/category_extraction/train_data.txt", "w") as f:
	for review in reviews:
		f.write(review + "\n")