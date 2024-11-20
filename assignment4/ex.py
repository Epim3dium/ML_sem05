import random as rnd
import numpy as np
import time
import matplotlib.pyplot as plt

def generate(n):
    return [0 if rnd.random() >= 0.5 else 1 for _ in range(n)]
def fitness(target, guess):
    return sum([1 - abs(target[i] - guess[i]) for i in range(len(target))])
def mutateOne(pattern):
    i = rnd.randrange(len(pattern))
    pattern[i] = 1 - pattern[i]
def mutateTillImprovement(target, guess, max_iters = 1000):
    if target == guess:
        return;
    original = guess

    fit = fitness(target, guess)
    cur_fit = fit

    for i in range(max_iters):
        mutateOne(guess)
        cur_fit = fitness(target, guess)
        if(fit < cur_fit):
            break
        guess = original

def sortParallel(key, value):
    new_list = []
    for x, y in zip(key, value):
        new_list.append((x, y))
    new_list = sorted(new_list, key=lambda element: -element[0])
    key, value = [], []
    for x, y in new_list:
        key.append(x)
        value.append(y)
    return key, value

def mutatedGuess(pattern, trainers=100, acceptance=0.3):
    n = len(pattern)
    cutoff = int(n * acceptance)
    to_generate = n - int(n * acceptance)
    guesses = [generate(n) for _ in range(trainers)]
    last_best = []
    i = 0
    while True:
        #calculating fitness
        fitnesses = [fitness(pattern, g) for g in guesses]
        fitnesses, guesses = sortParallel(fitnesses, guesses)
        #stagnation detected
        if last_best == guesses:
            return i#, fitnesses[0]
        last_best = guesses

        guesses = guesses[:cutoff]

        #mutating
        for ii in range(to_generate):
            copy = guesses[ii % cutoff].copy()
            mutateTillImprovement(pattern, copy)
            guesses.append(copy)
        i = i + 1



def randGuess(pattern):
    n = len(pattern)
    guess = generate(n)
    total_guesses = 1
    while(guess != pattern):
        total_guesses += 1
        guess = generate(n)
    return total_guesses

def guessInRange(min, max, guessFunc = randGuess, trials = 30):
    results = []
    for i in range(min, max):
        pattern = generate(i)
        iters_trial = []
        time_trial = []
        for _ in range(trials):
            start = time.time()
            iters_trial.append(guessFunc(pattern))
            end = time.time()
            time_trial.append(end - start)
        results.append({"bits" : i, "iters" : iters_trial, "time" : time_trial})
    return results
def ex11B():
    data = []
    lab= []
    times = []
    for result in guessInRange(1, 16):
        lab.append(result['bits'])
        data.append(result['iters'])
        times.append(sum(result['time']) / len(result['time']))
    plt.boxplot(data, tick_labels=lab)
    plt.show()
    plt.plot(times)
    plt.show()
def ex12():
    data = []
    lab= []
    times = []
    for result in guessInRange(4, 16, mutatedGuess):
        lab.append(result['bits'])
        data.append(result['iters'])
        times.append(sum(result['time']) / len(result['time']))
    plt.boxplot(data, tick_labels=lab)
    plt.show()
    plt.plot(times)
    plt.show()
ex12()
