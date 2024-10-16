from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn.utils import shuffle
import numpy as np
def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]
#
# # check which datasets can be imported
# list_available_datasets()

# import dataset
iris = fetch_ucirepo(id=53)
# alternatively: fetch_ucirepo(name='Heart Disease')

# access data
X = iris.data.features
y = iris.data.targets
print(X)
print(y)
X, y = shuffle(X, y)
y = y.reset_index(drop=True)
X = X.reset_index(drop=True)
print(X)
print(y)

delim = len(y) * 0.7
end = len(y)

# train model e.g. sklearn.linear_model.LinearRegression().fit(X, y)

# access metadata
# print(iris.metadata.uci_id)
# print(iris.metadata.num_instances)
# print(iris.metadata.additional_info.summary)

# access variable info in tabular format
# print(iris.variables.name)
def distance(a, b):
    sum = 0
    for attrib in X:
        sum = sum + (X[attrib][a] - X[attrib][b])**2
    return np.sqrt(sum)
k = 3
def print_props(idx):
    for a in X:
        print(a, X[a][idx])
def classify(idx):
    assert(idx >= delim)
    assert(idx < end)
    distances = []
    for i in range(0, int(delim)):
        distances.append([distance(idx, i), y['class'][i]])
    distances.sort()
    distances = distances[0:k]
    track = {}
    for d in distances:
        if d[1] not in track:
            track[d[1]] = 0
        track[d[1]] = track[d[1]] + 1
    winner = ""
    winner_count = 0
    for key, value in track.items():
        if(value > winner_count):
            winner_count = value
            winner = key
    return winner 
print(classify(delim + 1))




# for a in X:
#     print(X[a][0])
#     print(X[a][1])
