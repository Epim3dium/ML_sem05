#1
inputs = [[0, 0], [1, 0], [0, 1], [1, 1]]
#last one is bias
full_inputs = [i + [1] for i in inputs]
print("inputs: ", inputs)
expected_outputs = [0, 1, 1, 1]
print("expected outputs: ", expected_outputs)
#2
import numpy as np 
def sig(x):
    return 1/(1 + np.exp(-x))

def relu_prime(z):
    return 1 if z > 0.5 else 0
def activation_function(x):
    return relu_prime(x)

import random

#345
def getOutputFor(full_inputs, weights):
    outputs = []
    for input in full_inputs:
        sum = 0
        for i in range(len(input)):
            sum += weights[i] * input[i];
        outputs.append(activation_function(sum))
    return outputs;
def getWeightCorrection(full_inputs, error, alpha):
    dw = [0] * len(full_inputs[0])
    for i in range(len(error)):
        e = error[i]
        input = full_inputs[i]
        dw = [dw[ii] + e * alpha * input[ii] for ii in range(len(dw))] 
    return dw
def trainUntilConvergence(alpha, full_inputs, expected_outputs):
    weights = [(random.random() / 1.5) for _ in range(3)]
    outputs = [] 
    epochs_done = 0
    while outputs != expected_outputs:
        outputs = getOutputFor(full_inputs, weights)
        error = [expected_outputs[i] - outputs[i] for i in range(len(outputs))]
        dw = getWeightCorrection(full_inputs, error, alpha)
        for w in range(len(weights)):
            weights[w] += dw[w];
        epochs_done += 1
    return epochs_done

def ex456():
    epochs = 20
    alpha = 10e-3
    weights = [(random.random() / 1.5) for _ in range(3)]
    errors_recorded = [[] for _ in range(len(expected_outputs))]
    weights_recorded = [[] for _ in range(len(weights))]

    for _ in range(epochs):
        outputs = getOutputFor(full_inputs, weights)
        print("\tweights: ", weights)
        print("\toutputs: ", outputs)
        #3
        error = [expected_outputs[i] - outputs[i] for i in range(len(outputs))]
        for i in range(len(error)):
            errors_recorded[i].append(error[i])
        print("\terror: ", error)
        #4

        dw = getWeightCorrection(full_inputs, error, alpha)
        for w in range(len(weights)):
            weights[w] += dw[w];
        print("\td_weight: ", dw)
        for i in range(len(weights)):
            weights_recorded[i].append(weights[i])
    print("outputs: ", getOutputFor(full_inputs, weights))
    #6
    #a)
    import matplotlib.pyplot as plt
    colors = ['r', 'g', 'b', 'c']
    for i in range(len(errors_recorded) - 1):
        #offset by 0.01 to see the datapoints
        plt.plot([errors_recorded[i][ii] + i * 0.01 for ii in range(len(errors_recorded[i]))], 'x', color=colors[i])
    plt.legend(["error " + str(i + 1) for i in range(len(errors_recorded) - 1)])
    plt.ylim(-1.2, 1.2)
    plt.show()


    #b)
    for i in range(len(weights_recorded)):
        plt.plot(weights_recorded[i], 'x', color=colors[i])
    plt.legend(["weight 1", "weight 2", "weight 0"])
    plt.show()
    #b)
def ex6last():
    alpha = 10e-3 
    iterations = []
    for _ in range(30):
        iterations.append(trainUntilConvergence(alpha, full_inputs, expected_outputs))
    print("all iterations: ", iterations)
    print("average needed iterations:  ", sum(iterations) / len(iterations))
    import statistics
    print("standard dev of iterations: ", statistics.stdev(iterations))
def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]
def ex789():
    meanA = [ 3 , 3 ]
    covA = [ [ 1 , 0 ] , [ 0 , 1 ] ]
    meanB = [ -2.5 , -2.5]
    covB = [ [ 2.5 , 0 ] , [ 0 , 5 ] ]

    a = np.random.multivariate_normal (meanA, covA , 500).T
    b = np.random.multivariate_normal(meanB, covB , 500). T
    #1 for A
    expected = np.array([1] * 500 + [0] * 500);
    bias = np.array([1] * 1000)
    c = np.concatenate((a, b) , axis=1 )
    c = c.T

    c, expected = unison_shuffled_copies(c, expected)

    c = c.T
    x = c[0]
    y = c[1]
    import matplotlib.pyplot as plt
    # plt.plot(x, y, 'x')
    # plt.axis('equal')
    # plt.show()
    epochs = 20
    alpha = 10e-3
    weights = [(random.random() / 2) for _ in range(3)]

    full_inputs = [np.concatenate((c.T[i], np.array([1.0]))) for i in range(len(c.T))]
    expected_outputs = expected
    # print(expected_outputs)
    # print(full_inputs)
    for _ in range(epochs):
        outputs = getOutputFor(full_inputs, weights)
        #3
        error = [expected_outputs[i] - outputs[i] for i in range(len(outputs))]
        #4

        dw = getWeightCorrection(full_inputs, error, alpha)
        for w in range(len(weights)):
            weights[w] += dw[w];
    outputs = getOutputFor(full_inputs, weights)
    #8
    ATPx = []
    ATPy = []
    BTNx = []
    BTNy = []
    FPx = []
    FPy = []
    FNx = []
    FNy = []
    for i in range(len(full_inputs)):
        if outputs[i] and expected[i]:
            ATPx.append(full_inputs[i][0])
            ATPy.append(full_inputs[i][1])
        if outputs[i] and not expected[i]:
            FPx.append(full_inputs[i][0])
            FPy.append(full_inputs[i][1])
        if not outputs[i] and not expected[i]:
            BTNx.append(full_inputs[i][0])
            BTNy.append(full_inputs[i][1])
        if not outputs[i] and expected[i]:
            FNx.append(full_inputs[i][0])
            FNy.append(full_inputs[i][1])

    plt.plot(ATPx, ATPy, 'x', color='c')
    plt.plot(BTNx, BTNy, 'x', color='b')
    plt.plot(FPx, FPy, 'x', color='r')
    plt.plot(FNx, FNy, 'x', color='m')
    plt.axis('equal')
    plt.legend(["A actual", "B actual", "A error", "B error"])
    plt.show()
    # error = [expected_outputs[i] - outputs[i] for i in range(len(outputs))]
    # print(error)
    print("\tprediction")
    print("\tA\tB")
    print("actual")
    print("A", "\t", len(ATPx), "\t", len(FNx))
    print("B", "\t", len(FPx), "\t", len(BTNx))
    # neat
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    conf_matrix = confusion_matrix(expected, outputs)
    print(conf_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['B', 'A'])  # reverse display labels
    disp.plot()
    # neat
    plt.show()
def ex10(num_runs):
    meanA = [ 3 , 3 ]
    covA = [ [ 1 , 0 ] , [ 0 , 1 ] ]
    meanB = [ -2.5 , -2.5]
    covB = [ [ 2.5 , 0 ] , [ 0 , 5 ] ]

    a = np.random.multivariate_normal (meanA, covA , 500).T
    b = np.random.multivariate_normal(meanB, covB , 500). T
    #1 for A
    expected = np.array([1] * 500 + [0] * 500);
    bias = np.array([1] * 1000)
    c = np.concatenate((a, b) , axis=1 )
    c = c.T

    c, expected = unison_shuffled_copies(c, expected)

    c = c.T
    epochs = 20
    alpha = 10e-3

    precision = []
    accuracy = []
    recall = []
    F1 = []

    full_inputs = [np.concatenate((c.T[i], np.array([1.0]))) for i in range(len(c.T))]
    expected_outputs = expected
    for _ in range(num_runs):
        weights = [(random.random() / 2) for _ in range(3)]
        for _ in range(epochs):
            outputs = getOutputFor(full_inputs, weights)
            #3
            error = [expected_outputs[i] - outputs[i] for i in range(len(outputs))]
            #4

            dw = getWeightCorrection(full_inputs, error, alpha)
            for w in range(len(weights)):
                weights[w] += dw[w];
        outputs = getOutputFor(full_inputs, weights)
        #8
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(full_inputs)):
            if outputs[i] and expected[i]:
                TP = TP + 1
            if outputs[i] and not expected[i]:
                FP = FP + 1
            if not outputs[i] and not expected[i]:
                TN = TN + 1
            if not outputs[i] and expected[i]:
                FN = FN + 1
        precision.append(TP/(TP + FP))
        recall.append(TP/(TP + FN))
        accuracy.append((TP + TN)/(TP + TN + FP + FN))
        F1.append((2*precision[-1]*recall[-1])/(precision[-1]+recall[-1]))
    print("accuracy: ", sum(accuracy) / len(accuracy))
    print("recall: ", sum(recall) / len(recall))
    print("precision: ", sum(precision) / len(precision))
    print("F1: ", sum(F1) / len(F1))
# ex456()
ex789()
# ex10(30)

