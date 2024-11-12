import random
import numpy
import time
import matplotlib.pyplot as plt

UP    = 'u'
DOWN  = 'd'
LEFT  = 'l'
RIGHT = 'r'
ACTIONS = [LEFT, RIGHT, UP, DOWN]

SIZE = 10
STATES = SIZE * SIZE + 1

def stateTransition(state, action):
    if action == UP and state > SIZE:
        state = state - SIZE
    if action == DOWN and state < SIZE * (SIZE - 1):
        state = state + SIZE
    if action == LEFT and state % SIZE != 1:
        state = state - 1
    if action == RIGHT and state % SIZE != 0 and state < 100:
        state = state + 1
    return state
def reward(state):
    if state == 100:
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
Q = [{a:0 for a in ACTIONS} for _ in range(STATES)]
def updateQ(a, y, state, action):
    new_state = stateTransition(state, action);
    immediate_reward = reward(new_state);
    max_future_Q = 0;
    for _, val in Q[new_state].items():
        max_future_Q = max(max_future_Q, val)
    Q[state][action] = (1 - a) * Q[state][action] + a * (immediate_reward + y * max_future_Q)

def episodeQ(a, y, n, start = 1):
    state = start
    for i in range(n):
        action = randAction()
        updateQ(a, y, state, action)
        state = stateTransition(state, action)
    return state
def pathfind(a, y, n, start=1):
    state = start
    for i in range(n):
        max_action = ''
        max_action_val = 0
        for key, val in Q[state].items():
            if(val > max_action_val):
                max_action_val = val
                max_action = key
        # updateQ(a, y, state, action)
        state = stateTransition(state, max_action)
        if(reward(state) > 0):
            return {"reward" : reward(state) / i, "epochs" : i}
    return {"reward" : 0, "epochs" : n}

def ex2Single(a, y, n):
    pathfind_time = []
    pathfind_iters= []
    pathfind_rewards= []
    stop_interval = 100
    prev_start = 1
    for _ in range(int(n/stop_interval)):
        prev_start = episodeQ(a, y, n, prev_start)

        #perform test
        start = time.time()
        result = pathfind(a, y, 1000)
        end = time.time()
        pathfind_time.append(end - start)
        pathfind_iters.append(result["epochs"])
        pathfind_rewards.append(result["reward"])
    return {"times" : pathfind_time, "epochs" : pathfind_iters, "rewards" : pathfind_rewards}
def ex2A(a, y, n, k):
    times = []
    iters = []
    rewards= []
    for _ in range(k):
        e = ex2Single(a, y, n);
        times = times + e["times"]
        iters= iters+ e["epochs"]
        rewards= rewards+ e["rewards"]



# ex1simulation(30)
ex2A(0.7, 0.99, 2000, 30)
for i in range(SIZE):
    for ii in range(SIZE):
        print(Q[i * 10 + ii], end='')
    print("\n")


