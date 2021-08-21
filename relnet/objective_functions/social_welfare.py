def SocialWelfare(g):
    """
    Computes the social welfare by taking the sum of each players reward
    :param S2V state graph: Takes the graph in S2V format
    each node will be a game player class with a self.reward attribute
    :return: SocialWelfare = sum of all player rewards
    """
    # Sum all player rewards
    SW = sum(g.rewards.values()) / g.max_reward
    # Return social welfare multiplied by the reward scale
    return SW


def MaxContribution(g):
    # Computes number of agents playing 1, normalised by the number of agents
    MC = len([i for i in g.node_labels if g.actions[i] == 1.]) / g.num_nodes
    return MC
