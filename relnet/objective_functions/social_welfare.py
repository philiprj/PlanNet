

class SocialWelfare(object):

    @staticmethod
    def compute(networkX_graph):
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

        return SW
