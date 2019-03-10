import numpy as np


class Environment:
    """A toy environment that provides a new state as well as a reward for a action provided by a robot.

    The environment consists of a 5x5 game board where the final states are associated with rewards
    The board contains three final states that end the simulation when being reached (see map.png)
    The robots action is only correctly executed with a p=0.8, the robot chooses an orthogonal action with p=0.1 each
    Certain actions, such as leaving the board or crossing predefined walls are prohibited
    If the robot decides on a prohibited action, he remains in his current state=coordinates
    For each action that the robot tries to take (successful or not), he receives a reward of -1 (=punishment)

    """

    def __init__(self):
        # initialize the board rewards for each coordinate
        height = 5
        width = 5
        board_rewards = np.full((height, width), 0)
        board_rewards[1, 1] = -100
        board_rewards[1, 2] = 50
        board_rewards[1, 3] = 100
        self.board_rewards = board_rewards

        # actions: 0=up,1=left, 2=down, 3=right
        # initialize the allowed actions for each coordinate (=state)
        allowed_actions = np.ones((5, 5, 4))  # coordinate1, coordinate2, action
        allowed_actions[:, 0, 1] = 0  # robot is not allowed to cross left border
        allowed_actions[:, 4, 3] = 0  # robot is not allowed to cross right border
        allowed_actions[0, :, 0] = 0  # robot is not allowed to cross upper border
        allowed_actions[4, :, 2] = 0  # robot is not allowed to cross lower border

        # add additional borders to harden the problem
        allowed_actions[1:4, :, 3] = 0
        allowed_actions[1:4, :, 1] = 0
        allowed_actions[0, 1, 2] = 0
        allowed_actions[1, 2, 2] = 0
        allowed_actions[0, 3, 2] = 0
        allowed_actions[1, 1, 0] = 0
        allowed_actions[2, 2, 0] = 0
        allowed_actions[1, 3, 0] = 0

        self.allowed_actions = allowed_actions

        # define states that lead to an end of a simulation
        self.final_states = [[1, 1], [1, 2], [1, 3]]

        # define the robots initial state
        self.robot_state = [0, 0]

    def reset(self):
        """Reset the environment's state."""

        self.robot_state = [0, 0]

    def receive_action(self, action):
        """Receives an action by the robot, updates its state accordingly and provides a new state as well as a reward.

        Args:
            action: action chosen by the robot.

        Returns:
            The new state of the robot (=coordinates)
            A reward for the action taken by the robot
            The state of the simulation (game_over or not)

        """

        state = self.robot_state

        # take action given by robot with p=0.8, take orthogonal actions with p=0.1 each
        # not part of the learning strategy (!= epsilon-greedy) but an additional difficulty given by the task
        randomized_action = {
            0: np.random.choice(4, 1, p=[0.8, 0.1, 0, 0.1]),
            1: np.random.choice(4, 1, p=[0.1, 0.8, 0.1, 0]),
            2: np.random.choice(4, 1, p=[0, 0.1, 0.8, 0.1]),
            3: np.random.choice(4, 1, p=[0.1, 0, 0.1, 0.8]),
        }[action].item()

        if self.allowed_actions[state[0], state[1], randomized_action]:  # if distorted action is allowed
            # calculate next state according to the distorted chosen action
            self.robot_state = {
                0: [state[0] - 1, state[1]],  # move up
                1: [state[0], state[1] - 1],  # move left
                2: [state[0] + 1, state[1]],  # move down
                3: [state[0], state[1] + 1],  # move right
            }[randomized_action]

            # if the robot lands in a final state: set flag to later on finish current simulation
            game_over = self.robot_state in self.final_states

            return self.robot_state, self.board_rewards[self.robot_state[0], self.robot_state[1]]-1, game_over

        else:  # if distorted action is not allowed: stay in current state
            return self.robot_state, -1, False
