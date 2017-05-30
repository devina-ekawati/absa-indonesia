actuals = []
with open("../../data/crf/CRF_test.txt", "r") as f:
	tokens = []
	for line in f:
		line = line.rstrip()
		if line:
			token = line.split()[2]
			if token != "ASPECT-B" and token != "ASPECT-I" and token != "O":
				tokens.append("O")
			else:
				tokens.append(line.split()[2])
		else:
			actuals.append(tokens)
			tokens = []

predictions = []
with open("../../data/crf/test_result.txt", "r") as f:
	tokens = []
	for line in f:
		line = line.rstrip()
		if line:
			tokens.append(line)
		else:
			predictions.append(tokens)
			tokens = []

target = ["ASPECT-B", "ASPECT-I", "O"]
confusion_matrix = [[], [], []]
for i in range(len(target)):
	for j in range(len(target)):
		confusion_matrix[i].append(0)

for actual, prediction in zip(actuals, predictions):
	for actual_token, prediction_token in zip(actual, prediction):
		confusion_matrix[target.index(actual_token)][target.index(prediction_token)] += 1\

print confusion_matrix