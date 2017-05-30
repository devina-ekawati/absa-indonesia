lines = []

with open("../../data/crf/report.txt") as f:
	for line in f:
		lines.append(line.rstrip())
indices = [i for i, x in enumerate(lines) if x == "L-BFGS terminated with the stopping criteria"]

precision = []
recall = []
f1 = []

for indice in indices:
	tokens = lines[indice-4].split()

	precision.append(float(tokens[4][1:-1]))
	recall.append(float(tokens[5][:-1]))
	f1.append(float(tokens[6][:-1]))

print "Precision: ", reduce(lambda x, y: x + y, precision) / len(precision)
print "Recall: ", reduce(lambda x, y: x + y, recall) / len(recall)
print "F1-score: ", reduce(lambda x, y: x + y, f1) / len(f1)