import csv

file_names = []

with open("data/reviews/mergeReviewFile.txt", "r") as f:
	for line in f:
		line = line.rstrip()
		token = line.split(";")
		file_names.append((token[0], token[1]))

for file_name in file_names:
	reviews1 = []
	reviews2 = []

	file1 = "data/reviews/1/" + file_name[0]
	file2 = "data/reviews/2/" + file_name[1]

	with open(file1, "rb") as f1:
		for line in f1:
			line = line.rstrip()
			token = line.split(";")
			reviews1.append((token[0], token[1]))

	with open (file2, "rb") as f2:
		reader = csv.reader(f2)
		reviews2 = map(tuple, reader)

	result = []
	i = 0
	for review1 in reviews1:
		if (i > 1):
			result.append(review1[1])
		else:
			i += 1

	i = 0
	for review2 in reviews2:
		if (i > 0):
			if (review2[0] not in result):
				result.append(review2[0])
		else:
			i += 1


	with open("data/reviews/full/"+file_name[1], "w") as f:
		for item in result:
			f.write('"' + item + '"\n')

