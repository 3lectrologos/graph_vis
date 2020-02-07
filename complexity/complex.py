import os
import numpy as np
import matplotlib.pylab as plt
import pickle as pcl


def get_complexities(a, s, low, up):
    indUp = np.where(s <= up)[0]
    indUp = indUp[-1]
    indLow = np.where(s >= low)[0]
    indLow = indLow[0]
    nbrLinearRegions = indUp - indLow + np.sum(s[indLow] > low) + np.sum(s[indUp] < up) -np.sum((a[0]+a[1] == 0)&(indLow==0))
    if indLow > 0:
        absA = np.sqrt(np.sum((a[(indLow+np.sum(s[indLow] == low)):(indUp+1+np.sum(s[indUp] < up))])**2))
    else:
        absA = np.sqrt(np.sum((a[2:(indUp+1+np.sum(s[indUp] < up))])**2) + (a[0] + a[1])**2)
    pathLength = np.sum(np.array([np.sqrt(1+np.sum(a[1:(i+1)])**2)*(s[i]-s[i-1]) for i in range(indLow+1,indUp+1)])) + np.sqrt(1+a[0]**2)*max(s[0]-low,0) + np.sqrt(1+np.sum(a[1:(indUp+2)])**2)*max(up-s[indUp],0)
    return nbrLinearRegions, absA, pathLength


EPOCHS = [0, 10, 20, 30, 40, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000]


def process_dir(dirname):
    print('Processing', dirname)
    # Remove all existing .txt files
    allfiles = os.listdir(dirname)
    c1 = []
    c2 = []
    c3 = []
    for file in allfiles:
        if not file.endswith('.pcl'):
            continue
        comps = process_file(os.path.join(dirname, file))
        t1, t2, t3 = zip(*comps)
        c1.append(t1)
        c2.append(t2)
        c3.append(t3)
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    c3 = np.asarray(c3)
    m1 = np.mean(c1, axis=0)
    m2 = np.mean(c2, axis=0)
    m3 = np.mean(c3, axis=0)
    s1 = 2*np.std(c1, axis=0)/np.sqrt(c1.shape[0])
    s2 = 2*np.std(c2, axis=0)/np.sqrt(c2.shape[0])
    s3 = 2*np.std(c3, axis=0)/np.sqrt(c3.shape[0])
    return (m1, s1), (m2, s2), (m3, s3)


def process_file(filename):
    # Read data
    try:
        with open(os.path.join(filename), 'rb') as fin:
            aouts, souts, louts, x, y = pcl.load(fin)
    except:
        try:
            with open(os.path.join(filename), 'rb') as fin:
                aouts, souts, louts, gouts, x, y = pcl.load(fin)
        except:
            with open(os.path.join(filename), 'rb') as fin:
                aouts, souts, louts, gouts, _, x, y = pcl.load(fin)
    print(filename, '->', len(aouts))
    print(np.max([np.max(np.abs(aouts[i][1:-1])) for i in range(1, len(aouts))]))
    print('-----')
    comps = []
    for i in range(len(souts)):
        comps.append(get_complexities(aouts[i], souts[i], 0, 2))
    return comps


if __name__ == '__main__':
    # Which data sets to plot
    DIRS = ['FlatInitSinComplex', 'FlatInitSinComplexRegular']
    # Which complexity measures to plot
    COMPS = [1, 2]
    res = []
    allfiles = os.listdir('.')
    for file in allfiles:
        if os.path.isdir(file) and file in DIRS:
            res.append(process_dir(file))
    res = list(zip(*res))
    for c in COMPS:
        print('COMP {0} ========================'.format(c))
        for m, s in res[c]:
            xy = zip(EPOCHS, m)
            print(''.join(['({0},{1})'.format(x, y) for x, y in xy]))
            print('------')
