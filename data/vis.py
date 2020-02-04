import numpy as np
import matplotlib.pylab as plt
import pickle as pcl
import csv


fname = 'HeFlatRight2'
#               t     i     s      col   width
nodeformat = '{0:d} {1:d} {2:.5f} {3:d} {4:.1f}\n'
#              t1    i1    t2    i2   col1  col2  width1  width2
edgeformat = '{0:d} {1:d} {2:d} {3:d} {4:d} {5:d} {6:.1f} {7:.1f}\n'
slo = -2
shi = 2.8


def relu(x):
    return np.clip(x, a_min=0, a_max=None)


def flatnet(x, a, s):
    y = a[-1] + a[0]*relu(s[0]-x)
    for i in range(1, len(a)-1):
        y += a[i]*relu(x-s[i-1])
    return y


def trans(t, tlo, thi):
    return (t-tlo)/(thi-tlo)


def xy2str(x, y):
    xy = list(zip(x, y))
    xy.sort(key=lambda x: x[0])
    fstr = ''.join(['({0:.4f},{1:.4f})'.format(x, y) for x, y in xy])
    return fstr


if __name__ == '__main__':
    with open(fname + '.pcl', 'rb') as fin:
        aouts, souts, louts, x, y = pcl.load(fin)
    whi = np.max([np.max(np.abs(aouts[i][1:-1])) for i in range(1, len(aouts))])
    wlo = 0
    nodestring = ''
    edgestring = ''
    maxi = 9
    with open('../' + fname + '_data.txt', 'w') as fout:
        xy = sorted(list(zip(x[0], y[0])), key=lambda x: x[0])
        writer = csv.writer(fout, delimiter=',')
        writer.writerows(xy)
    for i in range(1, maxi):
        #fx = np.linspace(slo, shi, 1000)
        fx = souts[i]
        fy = flatnet(fx, aouts[i], souts[i])
        with open('../' + fname + '_xy_' + str(i) + '.txt', 'w') as fout:
            writer = csv.writer(fout, delimiter=',')
            writer.writerows(zip(fx, fy))
        for j, s in enumerate(souts[i]):
            col = 8
            if s < slo:
                s = slo
                col = 0
            elif s > shi:
                s = shi
                col = 0
            st = trans(s, slo, shi)
            w = aouts[i][j+1]
            if col > 0:
                if w > 0:
                    col = 8
                else:
                    col = 7
            if j == 0:
                col = 3
            wt = trans(np.fabs(w), wlo, whi)
            nodestring += nodeformat.format(i-1, j, st, col, wt)
            if i < maxi-1:
                edgestring += edgeformat.format(i-1, j, i, j, col, 8, 0.5, 0.5)
    with open('../' + fname + '_nodes.txt', 'w') as nout:
        nout.write(nodestring)
    with open('../' + fname + '_edges.txt', 'w') as nout:
        nout.write(edgestring)
    with open('../' + fname + '_times.txt', 'w') as nout:
        timestring = ''
        for i in range(1, maxi):
            timestring += '1000\n'
        nout.write(timestring)
