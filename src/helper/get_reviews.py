def get_reviews(filename):
	reviews = []
	review = ""
	with open(filename, 'r') as f:
		for line in f:
			line = line.rstrip()
			if line:
				token = line.split()
				review += token[0] + " "
			else :
				reviews.append(review[:-1])
				review = ""
				
	return reviews

if __name__ == '__main__':
	reviews = get_reviews("../../data/crf/CRF_test.txt")
	with open("../../CRF_test_reviews.txt", "w") as f:
		for review in reviews:
			f.write(review + " .\n")