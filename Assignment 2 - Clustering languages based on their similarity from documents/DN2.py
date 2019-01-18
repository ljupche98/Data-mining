import os
import glob
import math
import random
import numpy as np
from unidecode import unidecode

# text is a string, k is length of the substring
def get_substring(text, k = 3):
    for i in range(len(text) - k + 1):
        yield text[i : i + k]

# u and v are maps such that u[x] = # occurrences of string x in u
def get_similarity(u, v, m = None):
    return sum((u[w] * v[w]) for w in (set(u.keys()) & set(v.keys()))) / (math.sqrt(sum((w ** 2) for w in u.values())) * math.sqrt(sum((w ** 2) for w in v.values())))

def fill_empty_dict(x):
    ret = {}

    for y in x:
        ret[y] = 0

    return ret

def get_naive_dist(u, v):
    w = set(u.keys()) | set(v.keys())

    nu = {}
    nv = {}
    for x in w:
        nu[x] = 0
        nv[x] = 0

        if x in u.keys():
            nu[x] = u[x]

        if x in v.keys():
            nv[x] = v[x]

    total = sum((nu[x] * nv[x]) for x in w)
    divide = math.sqrt(sum((nu[x] ** 2) for x in w)) * math.sqrt(sum((nv[x] ** 2) for x in w))
    return total / divide

# generate a set of k random distinct integers in the range [l, r]
def get_random_numbers(l, r, k):
    assert r - l + 1 >= k

    ret = set()
    while len(ret) < k:
        x = random.randint(l, r)
        ret.add(x)

    return ret

def count(x):
    ret = {}
    keys = set()

    for y in x:
        if y not in keys:
            keys.add(y)
            ret[y] = 0

        ret[y] += 1

    return ret

def calc_total_cost(N, dist, medoids):
    ret = 0

    for x in range(N):
        cur = 1

        if medoids[x] == False:
            for y in range(N):
                if medoids[y] == True:
                    cur = min(cur, dist[x, y])

        ret += cur

    return ret

def k_medoids_cluster(N, dist, names, k = 5):
    medoids = [False for _ in range(N)]
    rnd = get_random_numbers(0, N - 1, k)

    for x in rnd:
        medoids[x] = True

    medoids = np.array(medoids)

    update = True
    last = calc_total_cost(N, dist, medoids)

    while update:
        update = False

        for x in range(N):
            if medoids[x] == False:
                for y in range(N):
                    if medoids[y] == True:
                        medoids[y] = False
                        medoids[x] = True

                        cur = calc_total_cost(N, dist, medoids)

                        if cur < last:
                            update = True
                            last = cur
                            break
                        else:
                            medoids[y] = True
                            medoids[x] = False

            if update:
                break

    silh = []
    for x in range(N):
        f = s = 1

        for y in range(N):
            if medoids[y] == True:
                if dist[x, y] < f:
                    s = f
                    f = dist[x, y]
                elif dist[x, y] < s:
                    s = dist[x, y]

        silh.append((s - f) / max(f, s))

    avg = 0
    for x in silh:
        avg += x
    avg /= N

    """
    ln = ""
    for x in range(N):
        if medoids[x]:
           ln += names[x] + " "
    print(ln)

    for x in range(N):
        if medoids[x] == False:
            minv = -1

            for y in range(N):
                if medoids[y]:
                    if minv == -1 or dist[x, y] < dist[x, minv]:
                        minv = y

            print("%s - %s = %.5f" % (names[x], names[minv], dist[x, minv]))
    """

    return avg

def produce_csv_file(N, names, data):
    with open('data.csv', 'w') as fw:
        first_row = "Language"

        remaining = set()
        for x in data:
            for y in x:
                remaining.add(y)

        for x in remaining:
            first_row += ', "' + x.replace('\n', '\\n') + '"'

        fw.write('%s\n' % first_row)

        remaining = list(remaining)
        map = {}
        for i in range(len(remaining)):
            map[remaining[i]] = i

        for i in range(len(names)):
            cur = names[i]

            v = np.array([0 for x in remaining])

            x = data[i]
            for y in x:
                v[map[y]] = x[y]

            for j in range(len(v)):
                cur += ', ' + ('%d' % v[j])

            fw.write('%s\n' % cur)

def predict(names, data, pred_names, pred_data):
    N = len(names)
    K = len(pred_names)

    for i in range(K):
        u = list([(get_similarity(pred_data[i], data[j]) ** 2, j) for j in range(N)])
        u.sort(reverse = True)

        u = np.array(u)

        sum = 0
        for j in range(N):
            sum += u[j, 0]

        print("Prediction for %s: 1. %s (%.4f) 2. %s (%.4f) 3. %s (%.4f)" % (pred_names[i].split('.')[0], names[int(u[0, 1])].split('.')[0], u[0, 0] / sum, names[int(u[1, 1])].split('.')[0], u[1, 0] / sum, names[int(u[2, 1])].split('.')[0], u[2, 0] / sum))

# program runs in about 30 seconds



# PART 1
names = np.array([os.path.basename(file) for file in glob.glob("lang/*")])
N = len(names)

inv_names = {}
for i in range(N):
    inv_names[names[i]] = i

data = np.array([count(list(get_substring('\n ' + unidecode(open(file, "rt", encoding = "utf8").read()).strip().replace('\n\n', '\n') + ' \n'))) for file in glob.glob("lang/*")])
dist = np.matrix([[(1 - get_similarity(data[x], data[y])) for y in range(N)] for x in range(N)])



# PART 2
avg_silh = []

import time
st = time.time()
for x in range(100):
    avg_silh.append(round(1e3 * k_medoids_cluster(N, dist, names)) / 1e3)
print("EXEC TIME: %s" % (time.time() - st))

import matplotlib.pyplot as plt
plt.hist(x = avg_silh, bins = 100, facecolor = 'k', alpha = 0.75)
plt.xlabel('Average silhouette distribution')
plt.ylabel('Number of occurrences')
plt.axis([0, 1, 0, 100])
plt.grid(True)
plt.show()




# PART 3
pred_names = np.array([os.path.basename(file) for file in glob.glob("pred/*")])

pred_inv_names = {}
for i in range(N):
    pred_inv_names[names[i]] = i

pred_data = np.array([count(list(get_substring('\n ' + unidecode(open(file, "rt", encoding = "utf8").read()).strip().replace('\n\n', '\n') + ' \n'))) for file in glob.glob("pred/*")])
predict(names, data, pred_names, pred_data)



# PART 4
# produces a CSV file that can be opened with Orange
# that way, we can make a hierarchical clustering on the data
produce_csv_file(N, names, data)



# PART 5
test_names = np.array([os.path.basename(file) for file in glob.glob("test/*")])
K = len(test_names)

test_inv_names = {}
for i in range(K):
    test_inv_names[test_names[i]] = i

test_data = np.array([count(list(get_substring('\n ' + unidecode(open(file, "rt", encoding = "utf8").read()).strip().replace('\n\n', '\n') + ' \n'))) for file in glob.glob("test/*")])
test_dist = np.matrix([[(1 - get_similarity(test_data[x], test_data[y])) for y in range(K)] for x in range(K)])


test_avg_silh = []

st = time.time()
for x in range(100):
    test_avg_silh.append(round(1e3 * k_medoids_cluster(K, test_dist, test_names)) / 1e3)
print("EXEC TIME FOR NEWSPAPER DATA: %s" % (time.time() - st))

plt.hist(x = test_avg_silh, bins = 100, facecolor = 'k', alpha = 0.75)
plt.xlabel('Average silhouette distribution based on newspaper data')
plt.ylabel('Number of occurrences')
plt.axis([0, 1, 0, 100])
plt.grid(True)
plt.show()


#   for j in range(K):
#       for i in range(N):
#           print("Similarity between %s - %s = %.5f" % (names[i], test_names[j], get_similarity(data[i], test_data[j])))