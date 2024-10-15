import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
def create_points():
    meanA = [ -3 , 3 ]
    covA = [ [ 1 , 0 ] , [ 0 , 0.5 ] ]
    meanB = [ 3 , -3]
    covB = [ [ 1 , 0 ] , [ 0 , 1 ] ]
    meanC = [ -1 , -2 ]
    covC = [ [ 0.5 , 0 ] , [ 0 , 0.5 ] ]

    a = np.random.multivariate_normal(meanA, covA , 100).T
    b = np.random.multivariate_normal(meanB, covB , 100). T
    c = np.random.multivariate_normal(meanC, covC , 100). T
    result = np.concatenate((a, b, c) , axis=1 )
    result = result.T

    np.random.shuffle(result)
    result = result.T
    x = result[0]
    y = result[1]

    f = open("dataset2.txt", "a")
    af = open("a_points2.txt", "a")
    bf = open("b_points2.txt", "a")
    cf = open("c_points2.txt", "a")
    f.truncate(0);
    af.truncate(0);
    bf.truncate(0);
    cf.truncate(0);
    for x, y in result.T:
        f.write(str(x) + ';' + str(y) + "\n")

    for x, y in a.T:
        af.write(str(x) + ';' + str(y) + "\n")
    for x, y in b.T:
        bf.write(str(x) + ';' + str(y) + "\n")
    for x, y in c.T:
        cf.write(str(x) + ';' + str(y) + "\n")
    af.close()
    bf.close()
    cf.close()
    f.close()
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1-y2)**2);
def clusterData(rx1, ry1, rx2, ry2, x, y):
    r1cluster = [[], []]
    r2cluster = [[], []]
    for i in range(1000):
        if distance(x[i], y[i], rx1, ry1) < distance(x[i], y[i], rx2, ry2):
            r1cluster[0].append(x[i])
            r1cluster[1].append(y[i])
        else:
            r2cluster[0].append(x[i])
            r2cluster[1].append(y[i])
    return [r1cluster, r2cluster];
    
def merge_until(x, y, cluster_count):
    active = set({})
    distances = []
    # first calculate distance from each to every node
    for i in range(len(x)):
        # and set active for all indexes
        active.add(i)
        for ii in range(i + 1, len(x)):
            distances.append([distance(x[i], y[i], x[ii], y[ii]), [i , ii]])
    next_idx = len(x)

    distances = sorted(distances, key=lambda x: x[0])
    # merge until number is ok
    while len(active) > cluster_count:
        [idx1, idx2] = distances[0][1];
        # if distance info contains retired node, delete it
        while(idx1 not in active or idx2 not in active):
            distances.pop(0)
            [idx1, idx2] = distances[0][1];
        avgx = (x[idx1] + x[idx2]) / 2.0
        avgy = (y[idx1] + y[idx2]) / 2.0
        #instead of removing verticies remove them from active set to not change indexes
        active.remove(idx1)
        active.remove(idx2)

        x.append(avgx)
        y.append(avgy)
        #calculate new distance only for the merged vertex
        for i in active:
            distances.append([distance(x[i], y[i], x[next_idx], y[next_idx]), [i , next_idx]])
        active.add(next_idx)

        next_idx = next_idx + 1

        # sort newly added distances
        distances = sorted(distances, key=lambda x: x[0])
    remainx = []
    remainy = []
    for i in active:
        remainx.append(x[i])
        remainy.append(y[i])
    return [remainx, remainy]
#
# create_points()
# create_points()
all = [[], []] 
with open('dataset2.txt', 'r') as file:
    for line in file:
        coords = line.split(';')
        all[0].append(float(coords[0]))
        all[1].append(float(coords[1]))
a = [[], []] 
with open('a_points2.txt', 'r') as file:
    for line in file:
        coords = line.split(';')
        a[0].append(float(coords[0]))
        a[1].append(float(coords[1]))
b = [[], []] 
with open('b_points2.txt', 'r') as file:
    for line in file:
        coords = line.split(';')
        b[0].append(float(coords[0]))
        b[1].append(float(coords[1]))
c = [[], []] 
with open('c_points2.txt', 'r') as file:
    for line in file:
        coords = line.split(';')
        c[0].append(float(coords[0]))
        c[1].append(float(coords[1]))
x = all[0]
y = all[1]
[rx, ry] = merge_until(x, y, 3)
plt.plot( a[0] , a[1] , 'x', color='y')
plt.plot( b[0] , b[1] , 'x', color='b')
plt.plot( c[0] , c[1] , 'x', color='g')
plt.plot( rx , ry , 'o', color='r')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.show()
