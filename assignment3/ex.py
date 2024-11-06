import random;
import numpy

UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3

SIZE = 10

def stateTransition(state, action):
    if action == UP and state > SIZE:
        state = state - SIZE
    if action == DOWN and state <= SIZE * (SIZE - 1):
        state = state + SIZE
    if action == LEFT and state % SIZE > 1:
        state = state - 1
    if action == RIGHT and state % SIZE != 0:
        state = state + 1
    return state
def reward(state):
    if state == 100:
        return 100;
    return 0;
def randAction():
    return [UP, DOWN, LEFT, RIGHT][random.randrange(4)]
def episode(n):
    state = 0
    for i in range(n):
        state = stateTransition(state, randAction())
        if(reward(state) > 0):
            return {"reward" : reward(state), "epochs" : i}
    return {"reward" : 0, "epochs" : n}
def simulation(k):
    epochs = []
    rewards = []
    for _ in range(k):
        e = episode(1000)
        epochs.append(e["epochs"])
        rewards.append(e["reward"])
    print("iter averag: ", sum(epochs) / len(epochs))
    print("iter stddev: ", numpy.std(epochs))

    print("reward averag: ", sum(rewards) / len(rewards))
    print("reward stddev: ", numpy.std(rewards))
simulation(30)


