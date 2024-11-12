#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import time
import seaborn as sns
import matplotlib.pyplot as plt

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [LEFT, RIGHT, UP, DOWN]

SIZE = 10
STATES = SIZE * SIZE

def stateTransition(state, action):
    if action == UP and state > SIZE:
        state -= SIZE
    elif action == DOWN and state < SIZE * (SIZE - 1):
        state += SIZE
    elif action == LEFT and state % SIZE != 0:
        state -= 1
    elif action == RIGHT and state % SIZE != 9:
        state += 1
    return state

def reward(state):
    return 100 if state == 99 else 0

def randAction():
    return ACTIONS[random.randrange(4)]

def episode(n):
    state = 0
    for i in range(n):
        state = stateTransition(state, randAction())
        if reward(state) > 0:
            return {"reward": reward(state) / i, "epochs": i}
    return {"reward": 0, "epochs": n}

def ex1simulation(k):
    epochs = []
    rewards = []
    times = []
    for _ in range(k):
        start = time.time()
        e = episode(1000)
        end = time.time()
        epochs.append(e["epochs"])
        rewards.append(e["reward"])
        times.append(end - start)
    print("iter average:", sum(epochs) / len(epochs))
    print("iter stddev:", (sum((x - sum(epochs) / len(epochs)) ** 2 for x in epochs) / len(epochs)) ** 0.5)
    print("reward average:", sum(rewards) / len(rewards))
    print("reward stddev:", (sum((x - sum(rewards) / len(rewards)) ** 2 for x in rewards) / len(rewards)) ** 0.5)
    print("time stddev:", (sum((x - sum(times) / len(times)) ** 2 for x in times) / len(times)) ** 0.5)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].boxplot(rewards)
    axs[0].set_title('Reward per step')

    axs[1].boxplot(epochs)
    axs[1].set_title('Number of iterations')

    axs[2].boxplot(times)
    axs[2].set_title('Time elapsed')

    plt.tight_layout()
    plt.show()

Q = [[0] * len(ACTIONS) for _ in range(STATES)]

def resetQ():
    for i in range(STATES):
        for a in ACTIONS:
            Q[i][a] = 0

def maxQ(state):
    return max(Q[state])

# ZMIANA: Dodano parametr `greed` do `maxQAction`, by wybierać losową akcję z prawdopodobieństwem 1 - greed.
def maxQAction(state, greed):  
    if random.random() > greed:
        return randAction()
    max_value = max(Q[state])
    max_actions = [a for a, val in enumerate(Q[state]) if val == max_value]
    return random.choice(max_actions)

def updateQ(a, y, state, action):
    new_state = stateTransition(state, action)
    immediate_reward = reward(new_state)
    max_future_Q = maxQ(new_state)
    Q[state][action] = (1 - a) * Q[state][action] + a * (immediate_reward + y * max_future_Q)

# ZMIANA: Dodano parametr `greed`, by kontrolować wybór akcji w funkcji `episodeQGreedy`.
def episodeQGreedy(a, y, n, greed, start=0):  
    state = start
    for i in range(n):
        action = maxQAction(state, greed)
        updateQ(a, y, state, action)
        state = stateTransition(state, action)
    return state

def pathfind(a, y, n, start=1):
    state = start
    for i in range(n):
        max_action = maxQAction(state, 1.0)  # ZMIANA: Dodano `greed=1.0`, aby wymusić wybór najlepszej akcji
        state = stateTransition(state, max_action)
        if reward(state) > 0:
            return {"reward": reward(state) / i, "epochs": i}
    return {"reward": 0, "epochs": n}

def ex2Single(a, y, n, episodeFunc):
    resetQ()
    pathfind_time = []
    pathfind_iters = []
    pathfind_rewards = []
    stop_interval = 100
    prev_start = 0
    for _ in range(int(n / stop_interval)):
        prev_start = episodeFunc(a, y, n, prev_start)
        if prev_start == 99:
            prev_start = 0

        start = time.time()
        result = pathfind(a, y, 1000)
        end = time.time()
        pathfind_time.append(end - start)
        pathfind_iters.append(result["epochs"])
        pathfind_rewards.append(result["reward"])

    plt.plot(pathfind_iters, marker='o', linestyle='-')
    plt.show()
    return {"times": pathfind_time, "epochs": pathfind_iters, "rewards": pathfind_rewards}

# ZMIANA: Dodano `ex2Greedy`, aby testować różne wartości `greed` i rysować wyniki
def ex2Greedy(a, y, n, k, greed_values):  
    times = []
    iters = []
    rewards = []
    for greed in greed_values:
        for _ in range(k):
            e = ex2Single(a, y, n, lambda a, y, n, start: episodeQGreedy(a, y, n, greed, start))
            times += e["times"]
            iters += e["epochs"]
            rewards += e["rewards"]

    plt.plot(iters, marker='o', linestyle='-')
    plt.title("Number of Iterations per Greed Level")
    plt.show()

    return {"times": times, "epochs": iters, "rewards": rewards}

# ZMIANA: Dodano `episodeQGreedyVarying`, aby stopniowo zwiększać `greed` od 0.3 do 1.0 w trakcie epizodu
def episodeQGreedyVarying(a, y, n, start=0):  
    state = start
    for i in range(n):
        greed = 0.3 + (i / n) * 0.7
        action = maxQAction(state, greed)
        updateQ(a, y, state, action)
        state = stateTransition(state, action)
    return state

# ZMIANA: Wartości `greed_values` dla różnych eksperymentów
greed_values = [0.2, 0.5, 0.9]
results = ex2Greedy(0.7, 0.99, 20000, 1, greed_values)

# ZMIANA: Test dla stopniowo rosnącego `greed`
varying_results = ex2Single(0.7, 0.99, 20000, episodeQGreedyVarying)

plt.figure(figsize=(10, 10))
data = [[maxQ(i + ii * SIZE) for i in range(SIZE)] for ii in range(SIZE)]
sns.heatmap(data, annot=True, cmap='viridis')
plt.title('Heatmap from 2D Array')
plt.show()


# In[ ]:




