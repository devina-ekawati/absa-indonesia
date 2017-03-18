lines = []

with open("../../data/train_CRF.txt", 'r') as f:
	sep=' '
	i = 1
	for line in f:
		print repr(line)
		line = line.rstrip()
		lines.append(line)

with open("../../data/train_CRF.txt", 'w') as f:
	for item in lines:
		f.write("%s\n" % item)

# with open("test.txt", 'r') as f:
# 	for line in f:
# 		line = line.rstrip()
# 		temp = line.split()
# 		print len(temp)