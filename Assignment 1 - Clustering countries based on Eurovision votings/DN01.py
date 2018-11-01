import math
import random
import matplotlib.pyplot as p
from itertools import combinations

def read_file(file_name):
    """
    Read and process data to be used for clustering.
    :param file_name: name of the file containing the data
    :return: dictionary with element names as keys and feature vectors as values
    """

    rank = dict([])
    rank[0] = rank[''] = 0
    for i in range(1, 9):
        rank[i] = i
    rank[10] = 9
    rank[12] = 10

    # country_range == indices of columns that contain information about country's voting
    county_range = range(16, 63)

    f = open(file_name, "rt", encoding = "latin1")

    # data contains every line of the input file
    import csv
    data = []
    for line in csv.reader(f):
        line_strip = [word.strip() for word in line]
        data.append(line_strip)

    # points are ranked
    for i in range(1, len(data)):
        line = data[i]

        for x in county_range:
            y = line[x]

            if y == '':
                y = '0'

            line[x] = rank[int(y)]

    # country is a list of countries that could have sometime been voted for
    country = []
    for i in range(1, len(data)):
        country.append(data[i][1])

    # removing duplicates from those countries and sorting them
    country = set(country)
    country = list(country)
    country.sort()

    # years contains all years that we have data about contest's results
    years = []
    for i in range(1, len(data)):
        years.append(data[i][0])

    # removing duplicates from those years and sorting them
    years = set(years)
    years = list(years)
    years.sort()

    # initializing a dictionary where voting_map_total[a, b] is a pair of the total number of points that country a
    # has given to country b, and the number of years that the country a has been voting for
    voting_map_total = dict([])
    for cnf in county_range:
        for cnt in country:
            voting_map_total[data[0][cnf], cnt] = 0, 0

    for yr in years:
        for cn in county_range:
            voted = False

            for ln in data:
                if ln[0] == yr:
                    if ln[cn] != 0:
                        voted = True
                        break

            if voted:
                for ln in data:
                    if ln[0] == yr:
                        x, y = voting_map_total[data[0][cn], ln[1]]
                        voting_map_total[data[0][cn], ln[1]] = x + ln[cn], y + 1

    # taking average of the votes - ranks

    voting_map_avg = dict([])
    for cnf in county_range:
        for cnt in country:
            voting_map_avg[data[0][cnf], cnt] = 0

    for x, y in voting_map_total:
        u, v = voting_map_total[x, y]

        if v != 0:
            voting_map_avg[x, y] = u / v

    # initializing the return vector for each country
    ret = dict()
    for i in county_range:
        v = []

        for x, y in voting_map_avg:
            if data[0][i] == x:
                v.append(voting_map_avg[x, y])

        ret[data[0][i]] = v
    #   print("Vector for ", data[0][i], " = ", v)

#   print(voting_map_total)
#   print(voting_map_avg)
#   print(ret)

    # create a file with the filtered data that can be used for Orange

#   with open('data-clear.csv', 'w') as fw:
#       title = "Country, "

#       for i in range(len(country)):
#           if i > 0:
#               title += ", "

#           title += country[i]

#       fw.write(title.strip() + "\n")
#       for x in ret.keys():
#           px = ""

#           for i in range(len(ret[x])):
#               if i > 0:
#                   px += ", "

#               px += "%7.3f" % ret[x][i]

#           fw.write("%s, %s\n" % (x.strip(), px.strip()))

    return ret

class HierarchicalClustering:
    def __init__(self, data):
        self.data = data
        self.clusters = [[name] for name in self.data.keys()]
        self.tree = dict([])
        self.tree_data = []

    def row_distance(self, r1, r2):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(self.data[r1], self.data[r2])))

    # get 'net' values of a cluster, meaning that a certain cluster will be represented
    # in other words, it removes nested lists
    def getNet(self, x):
        if type(x) != list:
            self.net.append(x)
            return

        for y in x:
            self.getNet(y)

    # get the 'net' value of a certain cluster
    def fnet(self, x):
        self.net = []
        self.getNet(x)
        return self.net

    def cluster_distance(self, c1, c2):
        c1 = self.fnet(c1)
        c2 = self.fnet(c2)

        d = 0
        for x in c1:
            for y in c2:
                d += self.row_distance(x, y)
        return d / (len(c1) * len(c2))

    def closest_clusters(self):
        d, p = min((self.cluster_distance(x, y), (x, y)) for x, y in combinations(self.clusters, 2))
        return d, p

    def run(self):
        n = len(self.clusters)

        # store information about merging - kind of 'pointers' for further needs
        for i in range(len(self.clusters)):
            self.tree[i] = -1, -1, -1
            self.tree_data.append(self.clusters[i])

        it = len(self.clusters)
        end_it = 2 * len(self.clusters) - 1
        while it < end_it:
            a, (x, y) = self.closest_clusters()

            self.clusters.remove(x)
            self.clusters.remove(y)
            self.clusters.append([x] + [y])

            xm = ym = -1
            for i in range(len(self.tree_data)):
                if self.tree_data[i] == x:
                    xm = i

                if self.tree_data[i] == y:
                    ym = i

            if xm == -1 or ym == -1:
                print("Failed to merge ", x, " (", xm, ") AND ", y, " (", ym, ")")
            else:
                self.tree[it] = xm, ym, a

            self.tree_data.append([x] + [y])
            it += 1

        #   print("SIZE = ", len(self.clusters), " ... REMOVE = ", x, " AND ", y, " ... ADD = ", (x + y))
        #   print(self.clusters)

        self.clusters = self.clusters[0]
        self.net_clusters = self.fnet(self.clusters)

    # print ASCII dendrogram
    def plot_tree_rec(self, x, level):
        if x == -1:
            return

        u, v, w = self.tree[x]

        if u == -1 or v == -1:
            if u == -1 and v == -1:
                print("    " * level, "----", self.tree_data[x][0])
            else:
                print("ERROR")
            return

        self.plot_tree_rec(u, 1 + level)
        print("    " * level, "----|")
        self.plot_tree_rec(v, 1 + level)

    def plot_tree(self):
        """
        Use cluster information to plot an ASCII representation of the cluster
        tree.
        """
        self.plot_tree_rec(len(self.tree_data) - 1, 0)

    def draw_dendrogram_rec(self, x, last, cur_color):
        if x == -1:
            return

        u, v, w = self.tree[x]

        if u == -1 or v == -1:
            target = self.tree_data[x][0]

            for i in range(len(self.net_clusters)):
                if target == self.net_clusters[i]:
                    p.plot([i, i], [0, last], cur_color)
                    return last, i
            return

    #   a = self.findMiddle(u)
    #   b = self.findMiddle(v)

    #   itc = 0
        cl = cur_color
    #   while cl == cur_color:
    #       cl = random.choice(self.colors)

    #       itc += 1
    #       if (itc > 5):
    #        print("Wow. So unlucky! x%d" % itc)

        c, a = self.draw_dendrogram_rec(u, w, cl)
        d, b = self.draw_dendrogram_rec(v, w, cl)

        p.plot([a, b], [w, w], cl)
        p.plot([a, a], [c, w], cl)
        p.plot([b, b], [d, w], cl)

        return w, (a + b) / 2.0

    def get_country_order(self, x):
        if x == -1:
            return

        u, v, w = self.tree[x]

        if u == -1 or v == -1:
            if u == -1 and v == -1:
                self.corder.append(self.tree_data[x][0])
            else:
                print("ERROR")
            return

        self.get_country_order(u)
        self.get_country_order(v)

    """
        def findMin(self, x):
            if x == -1:
                return 0
    
            u, v, w = self.tree[x]
    
            if u == -1 or v == -1:
                target = self.tree_data[x][0]
    
                for i in range(len(self.net_clusters)):
                    if target == self.net_clusters[i]:
                        return i
    
            return min(self.findMin(u), self.findMin(v))
    
        def findMax(self, x):
            if x == -1:
                return 0
    
            u, v, w = self.tree[x]
    
            if u == -1 or v == -1:
                target = self.tree_data[x][0]
    
                for i in range(len(self.net_clusters)):
                    if target == self.net_clusters[i]:
                        return i
    
            return max(self.findMax(u), self.findMax(v))
    
        def findMiddle(self, x):
            return (self.findMin(x) + self.findMax(x)) / 2.0
    """

    def draw_dendrogram(self):
        self.corder = []
        self.get_country_order(len(self.tree_data) - 1)

    #   self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.colors = ['k']
        self.draw_dendrogram_rec(len(self.tree_data) - 1, 100, random.choice(self.colors))
        p.bar(self.corder, len(self.corder) * [0])
        p.xticks(rotation = 'vertical')
        p.show()

if __name__ == "__main__":
    DATA_FILE = "eurovision-final.csv"
    hc = HierarchicalClustering(read_file(DATA_FILE))
    hc.run()
    hc.plot_tree()
    hc.draw_dendrogram()