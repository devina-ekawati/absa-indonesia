import csv

filtered_data = []

# with open("../../data/sentiment_extraction/train_data.csv", "rb") as f:
#     reader = csv.reader(f, delimiter=';', quotechar='"')
#     for row in reader:
#         if "neutral" not in row:
#             filtered_data.append(row)
#
# with open("../../data/sentiment_extraction/train_data_3.csv", "wb") as f:
#     writer = csv.writer(f, delimiter=';', quotechar='"')
#     for item in filtered_data:
#         writer.writerow(item)

data_1 = []
data_2 = []
with open("../../data/sentiment_extraction/test_data.csv", "rb") as f:
    reader = csv.reader(f, delimiter=';', quotechar='"')
    next(reader)
    for row in reader:
        data_1.append(row)

with open("../../data/sentiment_extraction/test_data_cumulative.csv", "rb") as f:
    reader = csv.reader(f, delimiter=';', quotechar='"')
    header = next(reader)
    for row in reader:
        data_2.append(row)

print len(data_1), len(data_2)

for item1, item2 in zip(data_1, data_2):
    if "neutral" not in item1:
        filtered_data.append(item2)

with open("../../data/sentiment_extraction/test_data_cumulative_2.csv", "wb") as f:
    writer = csv.writer(f, delimiter=';', quotechar='"')
    writer.writerow(header)
    for item in filtered_data:
        writer.writerow(item)