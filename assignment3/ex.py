import random;
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3

SIZE = 10

def stateTransition(state, action):
    if action == UP and sate > SIZE:
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
print(randAction())



