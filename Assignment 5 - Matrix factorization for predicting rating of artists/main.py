import random
import numpy as np


class RecommendationSystem(object):
    def readLearn(self, x):
        return [int(x[0]), int(x[1]), float(x[2])]

    def readTest(self, x):
        return [int(x[0]), int(x[1])]

    def fmap(self, x):
        ret = dict()
        for i in range(len(x)):
            ret[x[i]] = i
        return ret

    def __init__(self, train_data, test_data, seed = 17):
        random.seed(seed)
        np.random.seed(seed)

        f = open(train_data); next(f)
        self.data = [self.readLearn(x.strip().split('\t')) for x in f]
        f = open(test_data); next(f)
        self.test = [self.readTest(x.strip().split('\t')) for x in f]

        self.map_users = self.fmap(list(set([x[0] for x in self.data]) | set([x[0] for x in self.test])))
        self.map_items = self.fmap(list(set([x[1] for x in self.data]) | set([x[1] for x in self.test])))

        self.data = [[self.map_users[x[0]], self.map_items[x[1]], x[2]] for x in self.data]
        self.test = [[self.map_users[x[0]], self.map_items[x[1]]] for x in self.test]

        self.L = len(self.data); self.T = len(self.test)
        self.N = len(self.map_users); self.M = len(self.map_items)

        self.sum = [0.0 for _ in range(self.N)]
        self.cnt = [0.0 for _ in range(self.N)]

        for x in self.data:
            self.sum[x[0]] = self.sum[x[0]] + x[2]
            self.cnt[x[0]] = self.cnt[x[0]] + 1

        self.avg = sum(x[2] for x in self.data) / self.L
        self.user_avg = [self.sum[i] / self.cnt[i] if self.cnt[i] > 0 else 0.0 for i in range(self.N)]

    def predict(self, P, Q, K, u, i):
        return sum(P[u][k] * Q[k][i] for k in range(K + 2))

    def split(self, x, perc = .7):
        n = len(x)
        f_n = int(perc * n)
        s_n = n - f_n
        perm = np.random.permutation(np.arange(n))[:f_n]
        f_d = [x[y] for y in perm]
        s_d = [x[y] for y in range(n) if y not in perm]
        return f_n, f_d, s_n, s_d

    def trainValidation(self, K, perc = .7, learning_rate = .01, reg_fact = .017, max_it = 500):
        l_n, l_d, v_n, v_d = self.split(self.data)
        v_n1, v_d1, v_n2, v_d2 = self.split(v_d, perc = .5)

        P = [[random.choice([0.01, -0.01]) for _ in range(K + 2)] for __ in range(self.N)]
        Q = [[random.choice([0.01, -0.01]) for _ in range(self.M)] for __ in range(K + 2)]

        for i in range(self.N):
            P[i][K] = 1

        for i in range(self.M):
            Q[K + 1][i] = 1

        it = 0; l_e1 = 1e9; l_e2 = 1e9; improve = True
        while it < max_it and improve:
            improve = False

            for x in l_d:
                u = x[0]; i = x[1]; r = x[2]
                r_pred = self.predict(P, Q, K, u, i)

                e_ui = (r - r_pred)

                for k in range(K):
                    Puk = P[u][k]; Qki = Q[k][i]
                    P[u][k] = Puk + learning_rate * (e_ui * Qki - reg_fact * Puk)
                    Q[k][i] = Qki + learning_rate * (e_ui * Puk - reg_fact * Qki)

                Puk = P[u][K]; Qki = Q[K][i]
                # P[u][K] = Puk + learning_rate * (e_ui * Qki - reg_fact * Puk)
                Q[K][i] = Qki + learning_rate * (e_ui * Puk - reg_fact * Qki)

                Puk = P[u][K + 1]; Qki = Q[K + 1][i]
                P[u][K + 1] = Puk + learning_rate * (e_ui * Qki - reg_fact * Puk)
                # Q[K + 1][i] = Qki + learning_rate * (e_ui * Puk - reg_fact * Qki)

            t_e1 = 0
            for x in v_d1:
                u = x[0]; i = x[1]; r = x[2]
                r_pred = self.predict(P, Q, K, u, i)
                e_ui = (r - r_pred) ** 2
                t_e1 = t_e1 + e_ui

            t_e2 = 0
            for x in v_d2:
                u = x[0]; i = x[1]; r = x[2]
                r_pred = self.predict(P, Q, K, u, i)
                e_ui = (r - r_pred) ** 2
                t_e2 = t_e2 + e_ui

            t_e1 = (t_e1 / v_n1) ** (1 / 2)
            t_e2 = (t_e2 / v_n2) ** (1 / 2)

            if t_e1 < l_e1 and t_e2 < l_e2:
                improve = True

            it = it + 1; l_e1 = t_e1; l_e2 = t_e2
            print("%3d %.7f %.7f" % (it, t_e1, t_e2))

    def train(self, K, learning_rate = .01, reg_fact = .017, max_it = 11):
        P = [[random.choice([0.01, -0.01]) for _ in range(K + 2)] for __ in range(self.N)]
        Q = [[random.choice([0.01, -0.01]) for _ in range(self.M)] for __ in range(K + 2)]

        for i in range(self.N):
            P[i][K] = 1

        for i in range(self.M):
            Q[K + 1][i] = 1

        it = 0; last_e = 1e9; improve = True
        while it < max_it and improve:
            improve = True

            total_e = 0
            for x in self.data:
                u = x[0]; i = x[1]; r = x[2]
                r_pred = self.predict(P, Q, K, u, i)

                e_ui = (r - r_pred)
                total_e = total_e + e_ui ** 2

                for k in range(K):
                    Puk = P[u][k]; Qki = Q[k][i]
                    P[u][k] = Puk + learning_rate * (e_ui * Qki - reg_fact * Puk)
                    Q[k][i] = Qki + learning_rate * (e_ui * Puk - reg_fact * Qki)

                Puk = P[u][K]; Qki = Q[K][i]
                #P[u][K] = Puk + learning_rate * (e_ui * Qki - reg_fact * Puk)
                Q[K][i] = Qki + learning_rate * (e_ui * Puk - reg_fact * Qki)

                Puk = P[u][K + 1]; Qki = Q[K + 1][i]
                P[u][K + 1] = Puk + learning_rate * (e_ui * Qki - reg_fact * Puk)
                #Q[K + 1][i] = Qki + learning_rate * (e_ui * Puk - reg_fact * Qki)

            total_e = (total_e / self.L) ** (1 / 2)

            if total_e < last_e:
                improve = True

            it = it + 1
            last_e = total_e
            #print("%3d %.7f" % (it, total_e))

        ftim = 0
        with open('out.txt', 'w') as fw:
            for x in self.test:
                y = self.predict(P, Q, K, x[0], x[1])

                if y < 0:
                    ftim = ftim + 1
                    y = self.user_avg[x[0]]

                if y > 10:
                    ftim = ftim + 1
                    y = min(y, 10)

                fw.write("%.11f\n" % y)
                print("%.11f" % y)

        #print(ftim / len(self.test))

x = RecommendationSystem('user_artists_training.dat', 'user_artists_test.dat', seed = 119)
#x.trainValidation(51)
x.train(51)
