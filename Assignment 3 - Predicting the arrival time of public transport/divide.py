import csv
import gzip
import datetime

WRITE = True
MAX_LINES = 100
FMT = "%Y-%m-%d %H:%M:%S.%f"

reader = csv.reader(gzip.open("train.csv.gz", "rt"), delimiter = "\t")
header = next(reader)

data = [[] for l in range(MAX_LINES)]
for l in reader:
    line = int(l[2])
    data[line].append(l)

for i in range(MAX_LINES):
    if len(data[i]) > 0:
        data[i] = sorted(data[i], key = lambda x: datetime.datetime.strptime(x[6], FMT))
        fw = open("clear/line-%d.csv" % i, "w")

        fw.write("\t".join(header) + "\n")
        for x in data[i]:
            fw.write("\t".join(x) + "\n")