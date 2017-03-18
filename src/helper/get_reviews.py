def get_reviews(filename):
	reviews = []
	review = ""
	with open(filename, 'r') as f:
		for line in f:
			line = line.rstrip()
			if line:
				token = line.split()
				if (token[0].isalnum()):
					review += token[0] + " "
			else :
				reviews.append(review[:-1])
				review = ""
				
	return reviews

if __name__ == '__main__':
	reviews = get_reviews("../../data/CRF_dataset.txt")
	with open("../../CRF_reviews.txt", "w") as f:
		for review in reviews:
			f.write(review + " .\n")