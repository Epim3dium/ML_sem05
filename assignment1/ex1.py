import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
def create_points():
    meanA = [ 3 , 3 ]
    covA = [ [ 1 , 0 ] , [ 0 , 1 ] ]
    meanB = [ -3 , -3]
    covB = [ [ 2 , 0 ] , [ 0 , 5 ] ]

    a = np.random.multivariate_normal (meanA, covA , 500).T
    b = np.random.multivariate_normal(meanB, covB , 500). T
    c = np.concatenate((a, b) , axis=1 )
    c = c.T

    np.random.shuffle(c)
    c = c.T
    x = c[0]
    y = c[1]

    f = open("dataset.txt", "a")
    af = open("a_points.txt", "a")
    bf = open("b_points.txt", "a")
    f.truncate(0);
    af.truncate(0);
    bf.truncate(0);
    print(c.T)
    for x, y in c.T:
        f.write(str(x) + ';' + str(y) + "\n")
    for x, y in a.T:
        af.write(str(x) + ';' + str(y) + "\n")
    for x, y in b.T:
        bf.write(str(x) + ';' + str(y) + "\n")
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
    
def pp1_3(rx1, ry1, rx2, ry2, x, y, alpha, iter_count):
    rx1pathSingleEpoch = []
    ry1pathSingleEpoch = []
    rx2pathSingleEpoch = []
    ry2pathSingleEpoch = []

    rx1path = []
    ry1path = []
    rx2path = []
    ry2path = []
    for iter in range(iter_count):
        for i in range(1000):
            if distance(x[i], y[i], rx1, ry1) < distance(x[i], y[i], rx2, ry2):
                rx1 = (1 - alpha) * rx1 + alpha * x[i];
                ry1 = (1 - alpha) * ry1 + alpha * y[i];
            else: 
                rx2 = (1 - alpha) * rx2 + alpha * x[i];
                ry2 = (1 - alpha) * ry2 + alpha * y[i];
            if iter == 0:
                rx1pathSingleEpoch.append(rx1)
                ry1pathSingleEpoch.append(ry1)
                rx2pathSingleEpoch.append(rx2)
                ry2pathSingleEpoch.append(ry2)
        rx1path.append(rx1)
        ry1path.append(ry1)
        rx2path.append(rx2)
        ry2path.append(ry2)
    [r1cluster, r2cluster] = clusterData(rx1, ry1, rx2, ry2, x, y)
    plt.plot( r1cluster[0] , r1cluster[1] , 'x', color='g')
    plt.plot( r2cluster[0] , r2cluster[1] , 'x', color='c')
    plt.plot(rx1path, ry1path, 'o', color='r') 
    plt.plot(rx2path, ry2path, 'o', color='b') 
    plt.plot([rx1, rx2], [ry1, ry2], 'o', color='y') 
    plt.legend(["r1's cluster", "r2's cluster", "r1's path", "r2's path", "centers of clusters"])
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.show()

    for line in plt.gca().lines:
        line.remove()
    plt.plot( r1cluster[0] , r1cluster[1] , 'x', color='g')
    plt.plot( r2cluster[0] , r2cluster[1] , 'x', color='c')
    plt.plot(rx1pathSingleEpoch, ry1pathSingleEpoch, 'o', color='r') 
    plt.plot(rx2pathSingleEpoch, ry2pathSingleEpoch, 'o', color='b') 
    plt.plot([rx1, rx2], [ry1, ry2], 'o', color='y') 
    plt.legend(["r1's cluster", "r2's cluster", "r1's path first passage", "r2's path first passage", "centers of clusters"])
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    print(str(rx1) + ' ' + str(ry1) + '\t' + str(rx2) + ' ' + str(ry2))
    plt.show()
def secondApproach(rx1, ry1, rx2, ry2, x, y, alpha, iter_count):
    rx1path = []
    ry1path = []

    rx2path = []
    ry2path = []
    rx1pathSingleEpoch = []
    ry1pathSingleEpoch = []
    rx2pathSingleEpoch = []
    ry2pathSingleEpoch = []
    for iter in range(iter_count):
        rx1path.append(rx1)
        ry1path.append(ry1)
        rx2path.append(rx2)
        ry2path.append(ry2)
        dx1 = 0
        dx2 = 0
        dy1 = 0
        dy2 = 0
        d1count = 0
        d2count = 0
        for i in range(1000):
            if distance(x[i], y[i], rx1, ry1) < distance(x[i], y[i], rx2, ry2):
                dx1 += x[i] - rx1
                dy1 += y[i] - ry1
                d1count += 1
            else: 
                dx2 += x[i] - rx2
                dy2 += y[i] - ry2
                d2count += 1
            if iter == 0:
                rx1pathSingleEpoch.append(rx1)
                ry1pathSingleEpoch.append(ry1)
                rx2pathSingleEpoch.append(rx2)
                ry2pathSingleEpoch.append(ry2)
        if(d1count != 0):
            rx1 += alpha / d1count * dx1
            ry1 += alpha / d1count * dy1

        if(d2count != 0):
            rx2 += alpha / d2count * dx2
            ry2 += alpha / d2count * dy2
    return [[[rx1, rx2], [ry1, ry2]], 
            [rx1path, ry1path], [rx2path, ry2path], 
            [rx1pathSingleEpoch, ry1pathSingleEpoch],
            [rx2pathSingleEpoch, ry2pathSingleEpoch]]
def pp7(repeats, n, x, y, alpha, iter_count):
    r1path = [[], []]
    r2path = [[], []]
    for _ in range(repeats):
        index1 = np.random.randint(0, n - 1)
        rx1 = x[index1]
        ry1 = y[index1]

        index2 = np.random.randint(0, n - 1)
        rx2 = x[index2]
        ry2 = y[index2]
        [[[rx1, rx2], [ry1, ry2]], _, _, _, _] = secondApproach(rx1, ry1, rx2, ry2, x, y, alpha, iter_count);
        if distance(3, 3, rx1, ry1) > distance(3, 3, rx2, ry2):
            rx1, ry1, rx2, ry2 = rx2, ry2, rx1, ry1
        r1path[0].append(rx1)
        r1path[1].append(ry1)
        r2path[0].append(rx2)
        r2path[1].append(ry2)
    plt.plot(r1path[0], r1path[1], 'o', color='r') 
    plt.plot(r2path[0], r2path[1], 'o', color='b') 
    plt.legend(["all positions of r1", "all positions of r2"])
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.show()

def pp4_6(rx1, ry1, rx2, ry2, a, b, x, y, alpha, iter_count):

    [[[rx1, rx2], [ry1, ry2]], 
            [rx1path, ry1path], [rx2path, ry2path], 
            [rx1pathSingleEpoch, ry1pathSingleEpoch],
            [rx2pathSingleEpoch, ry2pathSingleEpoch]] = secondApproach(rx1, ry1, rx2, ry2, x, y, alpha, iter_count);
    [r1cluster, r2cluster] = clusterData(rx1, ry1, rx2, ry2, x, y)
    # plt.plot( x , y , 'x' )
    # plt.plot( a[0] , a[1] , 'x', color='g')
    # plt.plot( b[0] , b[1] , 'x', color='c')
    # plt.xlim(-8, 8)
    # plt.ylim(-8, 8)
    # plt.show()


    for line in plt.gca().lines:
        line.remove()
    plt.plot( r1cluster[0] , r1cluster[1] , 'x', color='g')
    plt.plot( r2cluster[0] , r2cluster[1] , 'x', color='c')
    plt.plot(rx1path, ry1path, 'o', color='r') 
    plt.plot(rx2path, ry2path, 'o', color='b') 
    plt.plot([rx1, rx2], [ry1, ry2], 'o', color='y') 
    plt.legend(["r1's cluster", "r2's cluster", "r1's path", "r2's path", "centers of clusters"])
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.show()

    for line in plt.gca().lines:
        line.remove()
    # plt.plot(rx1pathSingleEpoch, ry1pathSingleEpoch, 'o', color='r') 
    # plt.plot(rx2pathSingleEpoch, ry2pathSingleEpoch, 'o', color='b') 
    # plt.plot([rx1, rx2], [ry1, ry2], 'o', color='y') 
    # plt.legend(["r1's cluster", "r2's cluster", "r1's path on the first epoch", "r2's path on the first epoch", "centers of clusters"])
    # plt.xlim(-8, 8)
    # plt.ylim(-8, 8)
    # plt.show()


    r1A = [[], []]
    r1B = [[], []]
    r2A = [[], []]
    r2B = [[], []]
    for i in range(1000):
        a_contains = -1 
        for ii in range(len(a[0])):
            if x[i] == a[0][ii] and y[i] == a[1][ii]:
                a_contains = ii
                break
        r1_contains = -1
        for ii in range(len(r1cluster[0])):
            if x[i] == r1cluster[0][ii] and y[i] == r1cluster[1][ii]:
                r1_contains = ii
                break
        if a_contains != -1:
            if r1_contains != -1:
                r1A[0].append(x[i])
                r1A[1].append(y[i])
            else:
                r2A[0].append(x[i])
                r2A[1].append(y[i])
        else:
            if r1_contains != -1:
                r1B[0].append(x[i])
                r1B[1].append(y[i])
            else:
                r2B[0].append(x[i])
                r2B[1].append(y[i])

    for line in plt.gca().lines:
        line.remove()

    plt.plot( r1A[0] , r1A[1] , 'x', color='g')
    plt.plot( r1B[0] , r1B[1] , 'o', color='r')
    plt.plot( r2A[0] , r2A[1] , 'x', color='y')
    plt.plot( r2B[0] , r2B[1] , 'x', color='c')
    plt.legend(["point closer to r1 lab 1", "points closer to r1 lab 2", "points closer to r2 lab 1", "points closer to r2 lab 2"])
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.show()
    print(str(rx1) + ' ' + str(ry1) + '\t' + str(rx2) + ' ' + str(ry2))

c = [[], []] 
with open('dataset.txt', 'r') as file:
    for line in file:
        coords = line.split(';')
        c[0].append(float(coords[0]))
        c[1].append(float(coords[1]))
a = [[], []] 
with open('a_points.txt', 'r') as file:
    for line in file:
        coords = line.split(';')
        a[0].append(float(coords[0]))
        a[1].append(float(coords[1]))
b = [[], []] 
with open('b_points.txt', 'r') as file:
    for line in file:
        coords = line.split(';')
        b[0].append(float(coords[0]))
        b[1].append(float(coords[1]))
n = 1000
x = c[0]
y = c[1]
# for i in range(n):
#     print(str(c[0][i]) + ' ' + str(c[1][i]))
index1 = np.random.randint(0, n - 1)
rx1 = x[index1]
ry1 = y[index1]

alpha = 0.1
index2 = np.random.randint(0, n - 1)
rx2 = x[index2]
ry2 = y[index2]
plt.plot(a[0], a[1], 'o', color='g') 
plt.plot(b[0], b[1], 'o', color='c') 
plt.legend(["cluster a", "cluster b"])
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.show()

# pp1_3(x[0], y[0], x[1], y[1], x, y, 5e-5, 100)
# pp4_6(x[0], y[0], x[1], y[1], a, b, x, y, 5e-2, 100)
# pp7(30, n, x, y, 5e-2, 100)
