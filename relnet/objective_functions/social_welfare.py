

def SocialWelfare(g):
    """
    Computes the social welfare by taking the sum of each players reward
    :param networkX_graph: Takes the graph in NetworkX format -
    each node will be a game player class with a self.reward attribute
    :return: SocialWelfare = sum of all player rewards
    """
    # Sum all player rewards
    SW = sum(g.rewards.values())
    # Return social welfare multiplied by the reward scale
    return SW
