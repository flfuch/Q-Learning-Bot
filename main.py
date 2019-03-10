#!/usr/bin/env python3
from environment import Environment
from robot import Robot

""" Epsilon-greedy implementation of the one-step Q-learning algorithm as described in:
http://incompleteideas.net/book/first/ebook/node65.html
Teaches a robot to navigate through an environment as depicted in map.png
Change the learning rate alpha, greediness epsilon and future discount factor gamma to define the robots learning behavior
see robot.py, environment.py and map.png for more information

"""


def main():
    rob = Robot(alpha=0.1, epsilon=1.0, gamma=0.98, epsilon_decrease=1/3000)
    env = Environment()

    # run 3050 simulations and only plot the last 50 simulations
    for i in range(3050):
        env.reset()
        rob.run_bot(env, plotting=(i >= 3000))


if __name__ == "__main__":
    main()
















