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
feat = iris.data.features
target = iris.data.targets


# train model e.g. sklearn.linear_model.LinearRegression().fit(X, y)

# access metadata
# print(iris.metadata.uci_id)
# print(iris.metadata.num_instances)
# print(iris.metadata.additional_info.summary)

# access variable info in tabular format
# print(iris.variables.name)
def distance(a, b, X):
    sum = 0
    for attrib in X:
        sum = sum + (X[attrib][a] - X[attrib][b])**2
    return np.sqrt(sum)
def classify(idx, k, X, y, delim, end):
    assert(idx >= delim)
    assert(idx < end)
    distances = []
    for i in range(0, int(delim)):
        distances.append([distance(idx, i, X), y['class'][i]])
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
def ex1(n):
    kvals = [3, 7, 11]
    correct = {}
    for k in kvals:
        correct[k] = []
    for _ in range(n):
        X, y = shuffle(feat, target)
        y = y.reset_index(drop=True)
        X = X.reset_index(drop=True)

        delim = len(y) * 0.7
        end = len(y)
        for k in kvals:
            predicted = [classify(i, k, X, y, delim, end) for i in range(int(delim) + 1, end)]
            expected = [y['class'][i] for i in range(int(delim) + 1, end)]
            cur_correct = 0
            for i in range(len(predicted)):
                if(predicted[i] == expected[i]):
                    cur_correct = cur_correct + 1
            correct[k].append(cur_correct)
    print(correct)
    #imports Matplotlib library and assigns shorthand 'plt'
    import matplotlib.pyplot as plt
    #imports Seaborn library and assigns shorthand 'sns'
    fig, axs = plt.subplots(1, len(kvals), figsize=(15, 5))
    idx = 0
    for key, value in correct.items():
        axs[idx].boxplot(value)
        axs[idx].set_title('k = ' + str(key))
        idx = idx + 1
    plt.tight_layout()
    plt.show()
def ex23():
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    X, y = shuffle(feat, target)
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)

    delim = len(y) * 0.7
    end = len(y)
    kvals = [3, 7, 11]
    for i in range(len(kvals)):
        k = kvals[i]
        predicted = [classify(i, k, X, y, delim, end) for i in range(int(delim) + 1, end)]
        expected = [y['class'][i] for i in range(int(delim) + 1, end)]
        conf_matrix = confusion_matrix(expected, predicted)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Iris-versicolor', 'Iris-setosa', 'Iris-virginica'])  # reverse display labels
        disp.plot()
        plt.title('k-val = ' + str(k))
    plt.show()

ex1(30)
ex23()
