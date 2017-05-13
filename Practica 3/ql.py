import random
import numpy as np
import matplotlib.pyplot as plt


# Environment size
width = 5
height = 16

# Actions
num_actions = 4

# *************************************** MY CODE ******************************************************

r_actions_list = {
    0: "UP",
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT"
}

# ******************************************************************************************************

actions_list = {"UP": 0,
                "RIGHT": 1,
                "DOWN": 2,
                "LEFT": 3
                }

actions_vectors = {"UP": (-1, 0),
                   "RIGHT": (0, 1),
                   "DOWN": (1, 0),
                   "LEFT": (0, -1)
                   }

# Discount factor
discount = 0.8

Q = np.zeros((height * width, num_actions))  # Q matrix
Rewards = np.zeros(height * width)  # Reward matrix, it is stored in one dimension

# State real number in our enviroment for a given specific matrix position (row and column params)
def getState(y, x):
    return y * width + x

# Matrix position of the state real number (state number param)
def getStateCoord(state):
    return int(state / width), int(state % width)

# Returns a vector with possible actions for the state (param)
def getActions(state):
    y, x = getStateCoord(state)
    actions = []
    if x < width - 1:
        actions.append("RIGHT")
    if x > 0:
        actions.append("LEFT")
    if y < height - 1:
        actions.append("DOWN")
    if y > 0:
        actions.append("UP")
    return actions

# Returns a specific (possible) action for the state (param)
def getRndAction(state):
    return random.choice(getActions(state))

# Return a random state in our enviroment
def getRndState():
    return random.randint(0, height * width - 1)


# Different Rewards values for non-final states
Rewards[4 * width + 3] = -100
Rewards[4 * width + 2] = -100
Rewards[4 * width + 1] = -100
Rewards[4 * width + 0] = -100
Rewards[9 * width + 4] = -100
Rewards[9 * width + 3] = -100
Rewards[9 * width + 2] = -100
Rewards[9 * width + 1] = -100

# Reward and state number for the final state
Rewards[3 * width + 3] = 1000
final_state = getState(3, 3)

print "\nTabla de recompensas"
print np.reshape(Rewards, (height, width))

# QLearning function
def qlearning(s1, a, s2):
    Q[s1][a] = Rewards[s2] + discount * max(Q[s2])
    return



# ******************************************************************************************************
# *************************************** MY CODE ******************************************************

# FUNCTIONS:

# Q matrix plot (Ya implementada)
def qMatrixPlot():
    s = 0
    ax = plt.axes()
    ax.axis([-1, width + 1, -1, height + 1])

    for j in xrange(height):

        plt.plot([0, width], [j, j], 'b')
        for i in xrange(width):
            plt.plot([i, i], [0, height], 'b')

            direction = np.argmax(Q[s])
            if s != final_state:
                if direction == 0:
                    ax.arrow(i + 0.5, 0.75 + j, 0, -0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
                if direction == 1:
                    ax.arrow(0.25 + i, j + 0.5, 0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
                if direction == 2:
                    ax.arrow(i + 0.5, 0.25 + j, 0, 0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
                if direction == 3:
                    ax.arrow(0.75 + i, j + 0.5, -0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
            s += 1

        plt.plot([i + 1, i + 1], [0, height], 'b')
        plt.plot([0, width], [j + 1, j + 1], 'b')

    plt.show()

# Actions Number Plot
def showActionsNumber(allActionsNum):
    # Comment any next 5 lines to hide function in the plot
    plt.plot(range(0, episodes), allActionsNum[0], label='Explore')
    plt.plot(range(0, episodes), allActionsNum[1], label='Greedy')
    plt.plot(range(0, episodes), allActionsNum[2], label='E-Greedy 5')
    plt.plot(range(0, episodes), allActionsNum[3], label='E-Greedy 10')
    plt.plot(range(0, episodes), allActionsNum[4], label='E-Greedy 15')
    plt.legend(loc='upper center', shadow=True)

    # Axis limit
    #plt.xlim(0, 200)
    #plt.ylim(0, 200)

    # Axis color
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")

    plt.show()

# Exploratory mode
def exploratoty(episodes):
    # Actions in each episode
    actionNum = 0.0
    # Total actions (all episodes) for the average calculation
    totalActionNum = 0.0
    #Actions per episode
    episodeActions = []
    # Episodes
    for i in xrange(episodes):
        state = getRndState()
        while state != final_state:
            action = getRndAction(state)
            y = getStateCoord(state)[0] + actions_vectors[action][0]
            x = getStateCoord(state)[1] + actions_vectors[action][1]
            new_state = getState(y, x)
            qlearning(state, actions_list[action], new_state)
            state = new_state
            actionNum += 1
            totalActionNum += 1
        episodeActions.append(actionNum)
        actionNum = 0
    return totalActionNum, episodeActions


# E-greedy Mode (E is the explorations's probability)
# greedy if E = 0
def eGreedy(episodes, ePercent):
    # Actions in each episode
    actionNum = 0.0
    # Total actions (all episodes) for the average calculation
    totalActionNum = 0.0
    #Actions per episode
    episodeActions = []
    # Episodes
    for i in xrange(episodes):
        state = getRndState()
        while state != final_state:
            # Exploration
            if (random.randint(0,100) < ePercent):
                action = getRndAction(state)
            # Exploitation if its possible
            else:
                if Q[state][np.argmax(Q[state])] == 0:
                    action = getRndAction(state)
                else:
                    action = r_actions_list[np.argmax(Q[state])]
            # print state, action, Q[state]
            y = getStateCoord(state)[0] + actions_vectors[action][0]
            x = getStateCoord(state)[1] + actions_vectors[action][1]
            new_state = getState(y, x)
            qlearning(state, actions_list[action], new_state)
            state = new_state
            actionNum += 1
            totalActionNum += 1
        episodeActions.append(actionNum)
        actionNum = 0
    return totalActionNum, episodeActions


# OPERATIONS:

# Number of episode
episodes = 100;
# Array of actions per episode for each mode
allActionsNum = []

# Exploratory mode
actions=exploratoty(episodes)
allActionsNum.append(actions[1])
print "\nAcciones promedio exploratorio: ", actions[0]/episodes, "\n"
#print "\nTabla Q Exploratorio\n", Q
#qMatrixPlot()
Q = np.zeros((height * width, num_actions))  # Q matrix

# E-Greedy mode with ePercent = 0
actions=eGreedy(episodes,0)
allActionsNum.append(actions[1])
print "\nAcciones promedio greedy: ", actions[0]/episodes, "\n"
#print "\nTabla Q Greedy\n", Q
#qMatrixPlot()
Q = np.zeros((height * width, num_actions))  # Q matrix

# E-Greedy mode with ePercent = 5
actions=eGreedy(episodes,5)
allActionsNum.append(actions[1])
print "\nAcciones promedio E-greedy 5: ", actions[0]/episodes, "\n"
#print "\nTabla Q E-Greedy 5\n", Q
#qMatrixPlot()
Q = np.zeros((height * width, num_actions))  # Q matrix

# E-Greedy mode with ePercent = 10
actions=eGreedy(episodes,10)
allActionsNum.append(actions[1])
print "\nAcciones promedio E-greedy 10: ", actions[0]/episodes, "\n"
#print "\nTabla Q E-Greedy 10\n", Q
#qMatrixPlot()
Q = np.zeros((height * width, num_actions))  # Q matrix

# E-Greedy mode with ePercent = 15
actions=eGreedy(episodes,15)
allActionsNum.append(actions[1])
print "\nAcciones promedio E-greedy 15: ", actions[0]/episodes, "\n"
#print "\nTabla Q E-Greedy 15\n", Q
#qMatrixPlot()


showActionsNumber(allActionsNum)


# ******************************************************************************************************


# Para valores negativos (-X 0 0 0) la Q se actualiza igual... no hay que controlar nada en el argmax.
# Se pueden utilizar recompensas negativas (las usadas aqui) o parciales (positivas menores que la del estado final)