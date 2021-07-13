import self as self


class SocialWelfare(object):

    def __init__(self, reward_scale_multiplier=1.):
        """
        :param reward_scale_multiplier: Reward scale multiplier for scaling the total reward
        """
        self.reward_scale_multiplier = reward_scale_multiplier

    def compute(self, networkX_graph):
        """
        Computes the social welfare by taking the sum of each players reward
        :param networkX_graph: Takes the graph in NetworkX format -
        each node will be a game player class with a self.reward attribute
        :return: SocialWelfare = sum of all player rewards
        """
        SW = 0.
        # Loop over each player and get the reward
        for node in list(networkX_graph.nodes):
            SW += node.reward

        return SW * self.reward_scale_multiplier
