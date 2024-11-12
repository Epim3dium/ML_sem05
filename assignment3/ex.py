import random
import numpy
import time
import seaborn as sns
import matplotlib.pyplot as plt

UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3
ACTIONS = [LEFT, RIGHT, UP, DOWN]

SIZE = 10
STATES = SIZE * SIZE

def stateTransition(state, action):
    if action == UP and state > SIZE:
        state = state - SIZE
    if action == DOWN and state < SIZE * (SIZE - 1):
        state = state + SIZE
    if action == LEFT and state % SIZE != 0:
        state = state - 1
    if action == RIGHT and state % SIZE != 9:
        state = state + 1
    return state
def reward(state):
    if state == 99:
        return 100;
    return 0;
def randAction():
    return ACTIONS [random.randrange(4)]
def episode(n):
    state = 0
    for i in range(n):
        state = statetransition(state, randaction())
        if(reward(state) > 0):
            return {"reward" : reward(state) / i, "epochs" : i}
    return {"reward" : 0, "epochs" : n}
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
    print("iter averag: ", sum(epochs) / len(epochs))
    print("iter stddev: ", numpy.std(epochs))

    print("reward averag: ", sum(rewards) / len(rewards))
    print("reward stddev: ", numpy.std(rewards))
    print("reward stddev: ", numpy.std(times))
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].boxplot(rewards)
    axs[0].set_title('reward per step')

    axs[1].boxplot(epochs)
    axs[1].set_title('number of iterations')

    axs[2].boxplot(times)
    axs[2].set_title('time elapsed')

    plt.tight_layout()
    plt.show()
Q = [[0] * len(ACTIONS) for _ in range(STATES)]
def resetQ():
    for i in range(STATES):
        for a in ACTIONS:
            Q[i][a] = 0
def maxQ(state):
    max_future_Q = 0;
    for val in Q[state]:
        max_future_Q = max(max_future_Q, val)
    return max_future_Q
def maxQAction(state):
    max_future_Q = 0;
    action = 0
    max_action = 0
    for val in Q[state]:
        if val == max_future_Q and random.random() > 0.5:
            max_action = action
        if val > max_future_Q:
            max_future_Q = val
            max_action = action
        action = action + 1
    return max_action

def updateQ(a, y, state, action):
    new_state = stateTransition(state, action);
    immediate_reward = reward(new_state);
    max_future_Q = maxQ(new_state);
    Q[state][action] = (1 - a) * Q[state][action] + a * (immediate_reward + y * max_future_Q)

def episodeQRand(a, y, n, start = 0):
    state = start
    for i in range(n):
        action = randAction()
        updateQ(a, y, state, action)
        state = stateTransition(state, action)
    return state
def episodeQMax(a, y, n, start = 0):
    state = start
    for i in range(n):
        action = maxQAction(state)
        updateQ(a, y, state, action)
        state = stateTransition(state, action)
    return state
def pathfind(a, y, n, start=1):
    state = start
    for i in range(n):
        max_action = maxQAction(state)
        # updateQ(a, y, state, action)
        state = stateTransition(state, max_action)
        if(reward(state) > 0):
            return {"reward" : reward(state) / i, "epochs" : i}
    return {"reward" : 0, "epochs" : n}

def ex2Single(a, y, n, episodeFunc):
    resetQ()
    pathfind_time = []
    pathfind_iters= []
    pathfind_rewards= []
    stop_interval = 100
    prev_start = 0
    for _ in range(int(n/stop_interval)):
        prev_start = episodeFunc(a, y, n, prev_start)
        if prev_start == 99:
            prev_start = 0

        #perform test
        start = time.time()
        result = pathfind(a, y, 1000)
        end = time.time()
        pathfind_time.append(end - start)
        pathfind_iters.append(result["epochs"])
        pathfind_rewards.append(result["reward"])
    plt.plot(pathfind_iters, marker='o', linestyle='-')
    plt.show()
    return {"times" : pathfind_time, "epochs" : pathfind_iters, "rewards" : pathfind_rewards}
def ex2A(a, y, n, k):
    times = []
    iters = []
    rewards= []
    for _ in range(k):
        e = ex2Single(a, y, n, episodeQRand);
        times = times + e["times"]
        iters= iters+ e["epochs"]
        rewards= rewards+ e["rewards"]
def ex2B(a, y, n, k):
    times = []
    iters = []
    rewards= []
    for _ in range(k):
        e = ex2Single(a, y, n, episodeQMax);
        times = times + e["times"]
        iters= iters+ e["epochs"]
        rewards= rewards+ e["rewards"]



# ex1simulation(30)
ex2B(0.7, 0.99, 20000, 1)
plt.figure(figsize=(10, 10))
data= [[maxQ(i + ii * SIZE) for i in range(SIZE)] for ii in range(SIZE)]
print(data)
sns.heatmap(data, annot=True, cmap='viridis')
plt.title('Heatmap from 2D Array')
plt.show()

