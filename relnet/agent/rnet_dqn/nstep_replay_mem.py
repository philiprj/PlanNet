import random
import numpy as np


class NstepReplaySubMemCell(object):
    """
    Create a memory buffer to sample experience from
    """
    def __init__(self, memory_size):
        """
        :param memory_size: Size of the replay buffer - newer experience will push out older
        """
        # Init parameters
        self.memory_size = memory_size
        self.actions = [None] * self.memory_size
        self.rewards = [None] * self.memory_size
        self.states = [None] * self.memory_size
        self.s_primes = [None] * self.memory_size
        self.terminals = [None] * self.memory_size
        # Counters for sampling
        self.count = 0
        self.current = 0

    def add(self, s_t, a_t, r_t, s_prime, terminal):
        """
        Adds a new experience to the buffer
        :param s_t: State
        :param a_t: Action
        :param r_t: Reward
        :param s_prime: Next State
        :param terminal: Terminal bool
        """
        self.actions[self.current] = a_t
        self.rewards[self.current] = r_t
        self.states[self.current] = s_t
        self.s_primes[self.current] = s_prime
        self.terminals[self.current] = terminal
        # Increment counter and set the current position in the buffer
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def add_list(self, list_st, list_at, list_rt, list_sp, list_term):
        """
        Adds a list of experience in a loop
        """
        for i in range(len(list_st)):
            if list_sp is None:
                sp = (None, None, None, None)
            else:
                sp = list_sp[i]
            self.add(list_st[i], list_at[i], list_rt[i], sp, list_term[i])

    def sample(self, batch_size):
        """
        Samples a set of experience from the buffer
        :param batch_size: Sample batch size
        :return: list_st, list_at, list_rt, list_s_primes, list_term
        """
        # Check we have enough experience in memory
        assert self.count >= batch_size
        # Set lists for return
        list_st = []
        list_at = []
        list_rt = []
        list_s_primes = []
        list_term = []
        # Sample buffer randomly with replacement
        for i in range(batch_size):
            idx = np.random.randint(0, self.count - 1)
            list_st.append(self.states[idx])
            list_at.append(self.actions[idx])
            list_rt.append(float(self.rewards[idx]))
            list_s_primes.append(self.s_primes[idx])
            list_term.append(self.terminals[idx])
        return list_st, list_at, list_rt, list_s_primes, list_term


def hash_state_action(s_t, a_t):
    # Creates a hash integer for a given state action
    key = s_t[0]
    base = 179424673
    for e in s_t[1].directed_edges:
        key = (key * base + e[0]) % base
        key = (key * base + e[1]) % base
    if s_t[2] is not None:
        key = (key * base + s_t[2]) % base
    else:
        key = (key * base) % base
    
    key = (key * base + a_t) % base
    return key


class NstepReplayMemCell(object):
    def __init__(self, memory_size, balance_sample=False):
        self.sub_list = []
        self.balance_sample = balance_sample
        self.sub_list.append(NstepReplaySubMemCell(memory_size))
        if balance_sample:
            self.sub_list.append(NstepReplaySubMemCell(memory_size))
            self.state_set = set()

    def add(self, s_t, a_t, r_t, s_prime, terminal):

        if not self.balance_sample or r_t < 0:
            self.sub_list[0].add(s_t, a_t, r_t, s_prime, terminal)
        else:
            assert r_t > 0
            key = hash_state_action(s_t, a_t)
            if key in self.state_set:
                return
            self.state_set.add(key)
            self.sub_list[1].add(s_t, a_t, r_t, s_prime, terminal)
    
    def sample(self, batch_size):
        if not self.balance_sample or self.sub_list[1].count < batch_size:
            return self.sub_list[0].sample(batch_size)
        
        list_st, list_at, list_rt, list_s_primes, list_term = self.sub_list[0].sample(batch_size // 2)
        list_st2, list_at2, list_rt2, list_s_primes2, list_term2 = self.sub_list[1].sample(batch_size - batch_size // 2)
        
        return list_st + list_st2, list_at + list_at2, list_rt + list_rt2, list_s_primes + list_s_primes2, list_term + list_term2


class NstepReplayMem(object):
    def __init__(self, memory_size, n_steps, balance_sample=False):
        # Creates ones buffer for each step in the MDP for the total action
        self.mem_cells = []
        for i in range(n_steps - 1):
            self.mem_cells.append(NstepReplayMemCell(memory_size, False))
        self.mem_cells.append(NstepReplayMemCell(memory_size, balance_sample))

        self.n_steps = n_steps
        self.memory_size = memory_size

    def add(self, s_t, a_t, r_t, s_prime, terminal, t):
        # Adds to the buffer for specified step
        self.mem_cells[t].add(s_t, a_t, r_t, s_prime, terminal)

    def add_list(self, list_st, list_at, list_rt, list_sp, list_term, t):
        for i in range(len(list_st)):
            if list_sp is None:
                sp = (None, None, None, None)
            else:
                sp = list_sp[i]
            self.add(list_st[i], list_at[i], list_rt[i], sp, list_term[i], t)

    def sample(self, batch_size, t=None):
        if t is None:
            t = np.random.randint(self.n_steps)

        list_st, list_at, list_rt, list_s_primes, list_term = self.mem_cells[t].sample(batch_size)
        return t, list_st, list_at, list_rt, list_s_primes, list_term
