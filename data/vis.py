import os
import numpy as np
import matplotlib.pylab as plt
import pickle as pcl
import csv
import yaml


#               t     i     s      col   width
nodeformat = '{0:d} {1:d} {2:.5f} {3:d} {4:.1f}\n'
#              t1    i1    t2    i2   col1  col2  width1  width2
edgeformat = '{0:d} {1:d} {2:d} {3:d} {4:d} {5:d} {6:.1f} {7:.1f}\n'


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


def process_dir(dirname):
    # Remove all existing .txt files
    allfiles = os.listdir(dirname)
    for file in allfiles:
        if file.endswith('.txt'):
            os.remove(os.path.join(dirname, file))
    # Read and write options
    with open(os.path.join(dirname, 'options.yml'), 'r') as ymlstream:
        try:
            options = yaml.safe_load(ymlstream)
        except yamlYAMLError as exception:
            print(exception)
    ILIST = options['ILIST']
    TLIST = options['TLIST']
    #WMIN = options['WMIN']
    WMIN = 0.0001
    SLO = options['SLO']
    SHI = options['SHI']
    XLO = options['XLO']
    XHI = options['XHI']
    YLO = options['YLO']
    YHI = options['YHI']
    with open(os.path.join(dirname, 'options.txt'), 'w') as fout:
        writer = csv.writer(fout, delimiter=' ')
        writer.writerow([str(p) for p in [SLO, SHI, XLO, XHI, YLO, YHI]])
    # Read data
    try:
        with open(os.path.join(dirname, dirname + '.pcl'), 'rb') as fin:
            aouts, souts, louts, x, y = pcl.load(fin)
    except:
        try:
            with open(os.path.join(dirname, dirname + '.pcl'), 'rb') as fin:
                aouts, souts, louts, gouts, x, y = pcl.load(fin)
        except:
            with open(os.path.join(dirname, dirname + '.pcl'), 'rb') as fin:
                aouts, souts, louts, gouts, _, x, y = pcl.load(fin)
    print(dirname, '->', len(aouts))
    print(np.max([np.max(np.abs(aouts[i][1:-1])) for i in range(1, len(aouts))]))
    whi = 10#np.max([np.max(np.abs(aouts[i][1:-1])) for i in range(1, len(aouts))])
    wlo = 0
    nodestring = ''
    edgestring = ''
    with open(os.path.join(dirname, 'data.txt'), 'w') as fout:
        xy = sorted(list(zip(x[0], y[0])), key=lambda x: x[0])
        writer = csv.writer(fout, delimiter=',')
        writer.writerows(xy)
    nodeset = set()
    for i, si in enumerate(ILIST):
        if SLO < souts[si][0]:
            fxbefore = np.linspace(SLO, souts[si][0], 10)
        else:
            fxbefore = []
        if SHI > souts[si][-1]:
            fxafter = np.linspace(souts[si][-1], SHI, 10)
        else:
            fxafter = []
        fx = np.hstack((fxbefore, souts[si], fxafter))
        fy = flatnet(fx, aouts[si], souts[si])
        with open(os.path.join(dirname, 'xy_' + str(i) + '.txt'), 'w') as fout:
            writer = csv.writer(fout, delimiter=',')
            writer.writerows(zip(fx, fy))
        for j, s in enumerate(souts[si]):
            #if s < SLO:
            #    s = SLO
            #elif s > SHI:
            #    s = SHI
            if s < SLO or s > SHI:
                continue
            st = trans(s, SLO, SHI)
            w = aouts[si][j+1]
            if w > 0:
                col = 8
            else:
                col = 7
            wt = trans(np.fabs(w), wlo, whi)
            if wt > WMIN:
                nodeset.add((i, j))
                nodestring += nodeformat.format(i, j, st, int(louts[si][0][j]), wt)
            if False:
                curlabels = [int(x) for x in louts[si][1]]
                prevlabels = [int(x) for x in louts[si-1][1]]
                thislabel = curlabels[j]
                try:
                    prevj = prevlabels.index(thislabel)
                    if (i, j) in nodeset and (i-1, prevj) in nodeset:
                        edgestring += edgeformat.format(i-1, prevj, i, j, col, 8, 0.5, 0.5)
                except ValueError:
                    pass
    with open(os.path.join(dirname, 'nodes.txt'), 'w') as nout:
        nout.write(nodestring)
    with open(os.path.join(dirname, 'edges.txt'), 'w') as nout:
        nout.write(edgestring)
    with open(os.path.join(dirname, 'times.txt'), 'w') as nout:
        timestring = ''
        for i in ILIST:
            timestring += '{0}\n'.format(TLIST[i])
        nout.write(timestring)


if __name__ == '__main__':
    allfiles = os.listdir('.')
    for file in allfiles:
        if os.path.isdir(file):
            process_dir(file)
