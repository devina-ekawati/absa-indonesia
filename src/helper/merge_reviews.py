import csv
import os

reviews1 = []
reviews2 = []

path = "../../data/reviews/previous"
for filename in os.listdir(path):
	
	with open(path + "/" + filename, "rb") as f:
		reader = csv.reader(f, delimiter=';', quotechar='"')
		for row in reader:
			if (row[1] != "text"):
				reviews1.append(row[1])

# print reviews1

file2 = "../../data/reviews/collective.csv"
with open (file2, "rb") as f2:
	reader = csv.reader(f2, delimiter=',', quotechar='"')
	for row in reader:
		if (row[0] != "content"):
			reviews2.append(row[0])

result = []
for review1 in reviews1:
	result.append(review1)

for review2 in reviews2:
	if (review2 not in result):
		result.append(review2)


with open("../../data/reviews/reviews.txt", "w") as f:
	for item in result:
		f.write(item + '\n')

