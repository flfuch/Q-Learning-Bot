import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
plt.ion()


class Robot:
    """A Q-learning driven robot which learns to navigate through a provided environment based on received rewards.

    The robots calculates an optimal action based on a learned
    Q-table which lists the expected final reward for taking a certain action from a certain state

    """

    def __init__(self, alpha=0.1, epsilon=1.0, gamma=0.98, epsilon_decrease=1/2000):
        self.alpha = alpha  # declares how much each experience of a (state,action) should update its current Q-value
        self.epsilon = epsilon  # declares with what probability a random action should be chosen
        self.gamma = gamma  # declares what discount should be applied to future rewards
        self.epsilon_decrease = epsilon_decrease  # declares how much epsilon should be decreased after each simulation

        self.actions = [0, 1, 2, 3]  # the robots available actions: walk up(0), left(1), down(2) and right(3)
        self.plotting = False  # declares whether environment and robot state will be plotted for each movement step
        self.iteration = 0  # counts the number of simulations
        self.Q_table = np.full((5, 5, 4), -1.0)  # Qt estimates discounted final reward if starting with (state, action)
        self.game_over = False  # flag, declares if current simulation should stop (when next checked)
        self.total_reward = 0  # stores the total collected reward for each simulation

    def restart(self):
        """Restart the robot to be ready for the next simulation."""

        self.iteration += 1
        self.total_reward = 0
        self.game_over = False
        self.epsilon = max(self.epsilon-self.epsilon_decrease, 0)  # decreases exploration in favor of exploitation

    def max_reachable_q(self, state, allowed_actions):
        """Finds the maximal (by 1 action) reachable Q-value from a given state.

        Args:
            state: current state of the robot.
            allowed_actions: boolean array that determines for each state, action pair whether the given action is legal

        Returns:
            The highest out of the Q-values for all allowed actions on the given state

        """

        max_reachable_q = -float('inf')
        for action in self.actions:
            if allowed_actions[state[0], state[1], action]:
                if self.Q_table[state[0], state[1], action] > max_reachable_q:
                    max_reachable_q = self.Q_table[state[0], state[1], action]
        return max_reachable_q

    def update_bot(self, environment):
        """Chooses a next step and updates the Q-table according to the environments response.

        Args:
            environment (Environment): the current state of the environment.

        """

        state = environment.robot_state  # get the robot's current state from the environment

        take_best_action = np.random.choice([1, 0], p=[1-self.epsilon, self.epsilon])  # take best action with p=(1-eps)
        if take_best_action:
            action = np.argmax(self.Q_table[state[0], state[1], :])  # choose action with the highest Q-value
        else:
            action = np.random.random_integers(0, 3)  # choose random action

        new_state, new_reward, self.game_over = environment.receive_action(action)

        self.total_reward += new_reward

        # update Q_table based on the one-step Q-learning algorithm:
        # (source) http://incompleteideas.net/book/first/ebook/node65.html
        self.Q_table[state[0], state[1], action] += \
            self.alpha * \
            (new_reward +
             self.gamma*self.max_reachable_q(new_state, environment.allowed_actions) -
             self.Q_table[state[0], state[1], action])

        if self.plotting:
            self.plot_board(environment.board_rewards, new_state)

    def plot_board(self, board_rewards, robot_state):
        """Displays the current state of the board with matplotlib.

        Args:
            board_rewards: the rewards for landing in each state, provided by the environment.
            robot_state: current state of the robot

        Todo:
            visualize the uncrossable inner borders

        """

        board_pixels = board_rewards
        # normalize pixels to 0-255
        board_pixels = \
            ((board_pixels - np.amin(board_pixels)) * 255 / (np.amax(board_pixels) - np.amin(board_pixels))).astype(int)
        board_pixels[robot_state[0], robot_state[1]] = 300
        plt.imshow(board_pixels, interpolation='nearest', cmap=cm.RdYlGn)
        plt.show()
        plt.pause(0.1)  # show generated plot for 0.1 seconds before proceeding (blocking)

    def run_bot(self, environment, debug=False, plotting=False):
        """Runs one simulation of the robot on the current environment.

        Args:
            environment (Environment): the current state of the environment.
            debug (bool): whether to print debug information or not
            plotting (bool): declares whether this simulation should be plotted.

        """

        self.plotting = plotting
        self.restart()

        while not self.game_over:  # take actions until the robot reaches a final state
            self.update_bot(environment)

        print("Final reward at iteration " + str(self.iteration) + ": " + str(self.total_reward))

        if debug:
            print(self.epsilon)
            print(self.Q_table[:, :, 0])
            print(self.Q_table[:, :, 1])
            print(self.Q_table[:, :, 2])
            print(self.Q_table[:, :, 3])

        if self.plotting:
            plt.pause(1)  # show final position of bot for 1 second
