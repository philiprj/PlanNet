import random


class PGGPlayer:

    def __init__(self, player_id, first_action=None):
        """
        :param player_id: Unique identifier for the agent
        :param first_action: Sets the initial action, if None will pick a random actions
        """
        self.id = player_id
        # Sets the current reward for the agent
        self.reward = 0
        # If first_action is not defines then pick a random action
        if first_action is not None:
            self.action = first_action
        else:
            self.action = random.randint(0, 2)
        # Define agent histories
        self.action_history = []
        self.reward_history = []

    def act(self, neighbor_actions):
        """
        Defines the best response for the agent given the neighbors previous action
        :param neighbor_actions: List containing the actions of the neighbors {0, 1}
        :return: Action selected using best response
        """
        # If any neighboring agents are contributing then chose to defect
        if 1 in neighbor_actions:
            self.action = 0
        else:
            self.action = 1
        # Update the history
        self.action_history.append(self.action)
        return self.action

    def updated_reward(self, reward):
        """
        :param reward: reward given for the previous round
        """
        self.reward = reward
        self.reward_history.append(reward)
