lines = []

with open("../../data/crf/CRF_test.txt", 'r') as f:
	sep=' '
	i = 1
	for line in f:
		print repr(line)
		line = line.rstrip()
		lines.append(line)

with open("../../data/crf/CRF_test.txt", 'w') as f:
	for item in lines:
		f.write("%s\n" % item)

# with open("../../data/output1.conll", 'r') as f:
# 	for line in f:
# 		line = line.rstrip()
# 		lines.append(line)

# with open("../../data/output2.conll", "w") as f:
# 	for line in lines:
# 		tokens = line.split("\t")
# 		if (tokens[3] != "PUNCT"):
# 			f.write(line+"\n")
# 		f.write(line+"\n")

