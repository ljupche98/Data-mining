import os
import csv
import gzip
import numpy
import linear
import random
import datetime


def split_tuple(x):
    a = [u for u, _ in x]
    b = [v for _, v in x]
    return [a, b]


def getReg(x):
    return x[0]


def getDri(x):
    return x[1]


def getDep(x):
    return x[6]


def getArr(x):
    return x[8]


def getRoute(x):
    return tuple(x[2 : 5])


def MAE(u, v):
    assert len(u) == len(v)
    if len(u) == 0:
        return 0
    return sum(abs(u[i] - v[i]) for i in range(len(u))) / len(u)


def MSE(u, v):
    assert len(u) == len(v)
    return sum((u[i] - v[i]) ** 2 for i in range(len(u))) / len(u)

def R2(u, v):
    assert len(u) == len(v)
    if len(u) == 0:
        return 1
    avg = sum(x for x in v) / len(v)
    sx = sum((x - avg) ** 2 for x in v)
    if sx == 0:
        return 1
    return 1 - sum((x - avg) ** 2 for x in u) / sx


def readNet(path):
    reader = csv.reader(gzip.open(path, "rt"), delimiter = "\t"); next(reader)
    return [l for l in reader]


def readData(line):
    if os.path.isfile("clear/line-%d.csv" % line) == False:
        return []

    file = open("clear/line-%d.csv" % line, "r"); next(file)

    datax = [l.split("\n")[0].split("\t") for l in file]
    return datax
    data = []
    for l in datax:
        x = datetime.datetime.strptime(l[6], "%Y-%m-%d %H:%M:%S.%f")

    return data


def mapData(data):
    it = 0; s = set()
    fmp = dict(); imp = dict()

    for x in data:
        route = x

        if route not in s:
            s.add(route)
            fmp.update({route: it})
            imp.update({it: route})
            it += 1

    return fmp, imp


def mapTime(time, FMT, DEC):
    x = datetime.datetime.strptime(time, FMT)
    wd = x.weekday()

    if DEC:
        if x.day == 25 or x.day == 26:
            wd = 6

    return 48 * wd + 2 * x.hour + (0 <= x.minute < 30)


def timeDifference(x, y, FMT):
    return datetime.datetime.strptime(x, FMT) - datetime.datetime.strptime(y, FMT)


def rankData(u, v):
    su = set(u); mx = 1 + max(u)
    c_data = [(0, 0) for _ in range(mx)]

    for i in range(len(u)):
        x, y = c_data[u[i]]
        c_data[u[i]] = x + v[i], y + 1

    c_avg = []
    for x in su:
        a, b = c_data[x]
        c_avg.append((a / b, x))
    c_avg = sorted(c_avg)

    inv = [0 for _ in range(mx)]
    for i in range(len(c_avg)):
        x, y = c_avg[i]
        inv[y] = x, i

    return split_tuple([inv[x] for x in u]), inv


def learn(data, DEC):
    map_reg, inv_map_reg = mapData([getReg(l) for l in data])
    map_dri, inv_map_dri = mapData([getDri(l) for l in data])
    map_route, inv_map_route = mapData([getRoute(l) for l in data])

    N = len(map_route)
    line_tim = []
    line_route_tim = [[] for _ in range(N)]
    lm_data = [(0, 0) for _ in range(N)]
    for l in data:
        mp_route = map_route[getRoute(l)]
        dtx = timeDifference(getArr(l), getDep(l), FMT)

        line_tim.append(dtx.seconds)
        line_route_tim[mp_route].append(dtx.seconds)

        x, y = lm_data[mp_route]
        lm_data[mp_route] = x + dtx.seconds, y + 1

    avg_line = sum(line_tim) / len(line_tim)
    avg_route = [sum(line_route_tim[i]) / len(line_route_tim[i]) for i in range(N)]
    avg_data = [x / max(1, y) for x, y in lm_data]

    dr = [[] for _ in range(N)]
    dd = [[] for _ in range(N)]
    dt = [[] for _ in range(N)]
    dy = [[] for _ in range(N)]
    for l in data:
        mp_route = map_route[getRoute(l)]
        dr[mp_route].append(map_reg[getReg(l)])
        dd[mp_route].append(map_dri[getDri(l)])
        dt[mp_route].append(mapTime(getDep(l), FMT, DEC))
        dy[mp_route].append(timeDifference(getArr(l), getDep(l), FMT).seconds)

    # [MAPPED_ROUTE][X] ... X == 0 ? AVG : X == 1 ? RANK FOR THE I-TH EXAMPLE
    lm_reg = []
    lm_dri = []
    lm_tim = []

    mpx_reg = []
    mpx_dri = []
    mpx_tim = []

    for i in range(N):
        x, y = rankData(dr[i], dy[i])
        lm_reg.append(x)
        mpx_reg.append(y)

        x, y = rankData(dd[i], dy[i])
        lm_dri.append(x)
        mpx_dri.append(y)

        x, y = rankData(dt[i], dy[i])
        lm_tim.append(x)
        mpx_tim.append(y)

    sr = [len(set(dr[i])) for i in range(N)]
    sd = [len(set(dd[i])) for i in range(N)]
    st = [len(set(dt[i])) for i in range(N)]

    for i in range(N):
        for j in range(len(lm_reg[i][1])):
            lm_reg[i][1][j] /= sr[i]; lm_dri[i][1][j] /= sd[i]; lm_tim[i][1][j] /= st[i]

    models = []
    for i in range(N):
        #print("i = %2d %s" % (i, inv_map_route[i]))

        Y = numpy.array(dy[i])
        X = numpy.array([
                [
                    #lm_reg[i][0][j],
                    lm_reg[i][1][j],
                    #lm_dri[i][0][j],
                    lm_dri[i][1][j],
                    #lm_tim[i][0][j],
                    lm_tim[i][1][j], lm_tim[i][1][j] ** 2,
                    lm_reg[i][1][j] + lm_dri[i][1][j] + lm_tim[i][1][j]
                ]
                for j in range(len(dy[i]))
        ])

        lr = linear.LinearLearner(lambda_ = 17)
        models.append(lr(X, Y))

    return models, avg_line, map_route, map_reg, map_dri, mpx_reg, mpx_dri, mpx_tim, sr, sd, st


def predict(data, models, avg_line, map_route, map_reg, map_dri, mpx_reg, mpx_dri, mpx_tim, sr, sd, st, DEC):
    ans = [None for _ in range(len(data))]

    for j in range(len(data)):
        if ans[j] is None and int(data[j][2]) == LINE:
            ans[j] = avg_line

            if getRoute(data[j]) in map_route.keys():
                if getReg(data[j]) in map_reg.keys() and getDri(data[j]) in map_dri.keys():
                    x = mpx_reg[map_route[getRoute(data[j])]]
                    y = mpx_dri[map_route[getRoute(data[j])]]
                    z = mpx_tim[map_route[getRoute(data[j])]]

                    mpr = map_reg[getReg(data[j])]
                    mpd = map_dri[getDri(data[j])]
                    mpt = mapTime(getDep(data[j]), FMT, DEC)

                    if mpr >= len(x) or mpd >= len(y) or mpt >= len(z):
                        continue

                    x = x[mpr]
                    y = y[mpd]
                    z = z[mpt]

                    if x == 0 or y == 0 or z == 0:
                        continue

                    a, b = x
                    c, d = y
                    e, f = z

                    b /= sr[map_route[getRoute(data[j])]]
                    d /= sd[map_route[getRoute(data[j])]]
                    f /= st[map_route[getRoute(data[j])]]


                    v = numpy.array([
                        #a,
                        b,
                        #c,
                        d,
                        #e,
                        f, f ** 2,
                        b + d + f
                    ])

                    ans[j] = models[map_route[getRoute(data[j])]](v)

    return ans


def cross_validation(data, K, ln):
    FMT = "%Y-%m-%d %H:%M:%S.%f"
    N = len(data)
    random.shuffle(data)

    cum = [0]; q, r = divmod(N, K)
    for i in range(K):
        cum.append(cum[i] + q + (i < r))

    ret = 0
    for i in range(K):
        lrn = data[0 : cum[i]] + data[cum[i + 1] : N]
        tst = data[cum[i] : cum[i + 1]]
        ans = [None for _ in range(len(tst))]

        models, avg_line, map_route, map_reg, map_dri, mpx_reg, mpx_dri, mpx_tim, sr, sd, st = learn(lrn, False)
        ansx = predict(tst, models, avg_line, map_route, map_reg, map_dri, mpx_reg, mpx_dri, mpx_tim, sr, sd, st, False)

        it = 0; jt = 0
        while it < len(ans) and jt < len(ansx):
            while it < len(ans) and int(tst[it][2]) != LINE:
                it += 1

            if it < len(ansx):
                ans[it] = ansx[jt]
                it += 1; jt += 1

        ref = [timeDifference(getArr(tst[i]), getDep(tst[i]), FMT).seconds for i in range(len(tst))]
        ret += MAE(ans, ref) * len(ans)
        print("Line %2d | %d. iteration: MAE = %.5f | R2 = %.5f" % (ln, 1 + i, MAE(ans, ref), R2(ans, ref)))

    return ret

FMT = "%Y-%m-%d %H:%M:%S.%f"
pred_ds = readNet("test.csv.gz")
pred_line = [[] for _ in range(100)]
for l in pred_ds:
    pred_line[int(l[2])].append(l)

SUM = 0
TOTX = 0
final_ans = [None for _ in range(len(pred_ds))]

for LINE in range(100):
    data = readData(LINE)
    TOTX += len(data)
    if len(data) == 0:
        continue

    models, avg_line, map_route, map_reg, map_dri, mpx_reg, mpx_dri, mpx_tim, sr, sd, st = learn(data, True)
    ans = predict(pred_line[LINE], models, avg_line, map_route, map_reg, map_dri, mpx_reg, mpx_dri, mpx_tim, sr, sd, st, True)

    i = 0; j = 0
    while i < len(final_ans) and j < len(ans):
        while i < len(final_ans) and int(pred_ds[i][2]) != LINE:
            i += 1

        if i < len(final_ans):
            final_ans[i] = ans[j]
            i += 1; j += 1

print("AVG MAE = %.5f" % (SUM / TOTX))
fx = open("out.txt", "w")
for i in range(len(pred_ds)):
    fx.write(str(datetime.datetime.strptime(getDep(pred_ds[i]), FMT) + datetime.timedelta(seconds = final_ans[i])) + "\n")