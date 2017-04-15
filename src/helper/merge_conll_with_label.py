lines1 = []
lines2 = []
results = []

with open("../../data/crf/CRF_test.txt", "r") as f:
	for line in f:
		line = line.rstrip()
		if line:
			lines1.append(line)

with open("../../data/output_test.conll", "r") as f:
	i = 0
	for line in f:
		line = line.rstrip()

		if line:
			lines2.append(line)

print len(lines1)
print len(lines2)

i = 0
with open("../../data/output1_test1.conll", "w") as f:
	for line2 in lines2:
		tokens1 = lines1[i].split()
		tokens2 = line2.split()

		if tokens2[0] == "1":
			f.write("\n")
		if tokens1[2] == "ASPECT-B" or tokens1[2] == "ASPECT-I" or tokens1[2] == "O":
			f.write(line2 + "\t" + tokens1[2] + "\n")
		else:
			f.write(line2 + "\tO\n")

		i += 1
