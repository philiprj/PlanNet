import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from relnet.agent.rnet_dqn.q_net import NStepQNet, greedy_actions
from relnet.agent.rnet_dqn.nstep_replay_mem import NstepReplayMem
from relnet.agent.pytorch_agent import PyTorchAgent
from relnet.utils.config_utils import get_device_placement


class RNetDQNAgent(PyTorchAgent):
    # Defines the agent name and parameters
    algorithm_name = "rnet_dqn"
    is_deterministic = False
    is_trainable = True

    def __init__(self, environment):
        """
        :param environment: The network the agent is playing on
        """
        super().__init__(environment)

    def setup(self, options, hyperparams):
        """
        :param options: Agent options
        :param hyperparams: Training hyperparameters
        """
        # Sets up the network and general parameters
        super().setup(options, hyperparams)
        self.setup_nets()
        self.take_snapshot()

    def train(self, train_g_list, validation_g_list, max_steps, **kwargs):
        """
        Trains the network on the training graphs - checking on the validation graphs
        :param train_g_list: Training graphs (list)
        :param validation_g_list: Validation graphs (list)
        :param max_steps: Max number of training steps
        :param kwargs: Takes an arbitrary number of keyword arguments - doesn't seem to be used
        """
        # Sets up graphs and get indices for training
        self.setup_graphs(train_g_list, validation_g_list)
        self.setup_sample_idxes(len(train_g_list))
        # Set up the memory/history and training parameters
        self.setup_mem_pool(max_steps, self.hyperparams['mem_pool_to_steps_ratio'])
        self.setup_histories_file()
        self.setup_training_parameters(max_steps)
        # Creates a progress bar, unit is each training batch - for burn in phase
        pbar = tqdm(range(self.burn_in), unit='batch', disable=None)
        # Loops through batches
        for p in pbar:
            # With no grad - run simulation
            with torch.no_grad():
                self.run_simulation()
        # Create progress bar using main steps
        pbar = tqdm(range(max_steps + 1), unit='steps', disable=None)
        # Set Adam optimizer
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        # Loop training steps and run simulation
        for self.step in pbar:
            with torch.no_grad():
                self.run_simulation()
            # If at snapshot step then take a snapshot
            if self.step % self.net_copy_interval == 0:
                self.take_snapshot()
            # Check validation loss each step
            self.check_validation_loss(self.step, max_steps)
            """
            Sample memory batch
            cur_time: time step {0,1} for the action
            list_st: (graph state, selected node - none if no selection, banned actions - empty if no selection) 
            list_as: action selected at s
            list_rt: reward received after action selected - 0 if non-terminal 
            list_s_primes: s' - graph state after selected action  
            list_term: list of terminal state where no reward if received
            """
            cur_time, list_st, list_at, list_rt, list_s_primes, list_term = self.mem_pool.sample(
                batch_size=self.batch_size)
            # Get target tensor and move to GPU
            list_target = torch.Tensor(list_rt)
            if get_device_placement() == 'GPU':
                list_target = list_target.cuda()
            # List to store s' which excludes the terminal s states - these are given rewards instantly in list_target
            cleaned_sp = []
            nonterms = []
            # Loop through state list - if the next graph state in None then s in terminal state - so do not add
            for i in range(len(list_st)):
                # If list_term is not none - append cleaned s' and track index
                if not list_term[i]:
                    cleaned_sp.append(list_s_primes[i])
                    nonterms.append(i)
            # IF list is not empty
            if len(cleaned_sp):
                # Unzip to get banned values
                _, _, banned = zip(*cleaned_sp)
                # Get the next q values for each node in each graph using the target network
                _, q_t_plus_1, prefix_sum_prime = self.old_net((cur_time + 1) % 2, cleaned_sp, None)
                # Get the greedy actions using the q values, and noting the banned actions
                _, q_rhs = greedy_actions(q_t_plus_1, prefix_sum_prime, banned)
                # Update the targets to be the greedy q values
                list_target[nonterms] = q_rhs
            # Convert to torch tensor and n x 1 tensor
            list_target = Variable(list_target.view(-1, 1))
            # Feed to the main network to get the q values for the states and actions selected
            _, q_sa, _ = self.net(cur_time % 2, list_st, list_at)
            # Calculates the loss use MSE from predicted and target values
            loss = F.mse_loss(q_sa, list_target)
            # Zero grad, propagate backwards, and take step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Set values for progress bar for printing
            pbar.set_description('exp: %.5f, loss: %0.5f' % (self.eps, loss))
            # Checks if early stopping condition has been met
            should_stop = self.check_stopping_condition(self.step, max_steps)
            if should_stop:
                break

    def setup_nets(self):
        """
        Initialises the agent network for training
        """
        self.net = NStepQNet(self.hyperparams, num_steps=2)
        self.old_net = NStepQNet(self.hyperparams, num_steps=2)
        # Sends Network to GPU or CPU depending on what is available
        if get_device_placement() == 'GPU':
            self.net = self.net.cuda()
            self.old_net = self.old_net.cuda()
        # Restores previous model if selected
        if self.restore_model:
            self.restore_model_from_checkpoint()

    def setup_mem_pool(self, num_steps, mem_pool_to_steps_ratio):
        """
        Sets up a replay buffer for learning from past experience
        :param num_steps: Number of steps for the lookahead
        :param mem_pool_to_steps_ratio: Ratio of the size of memory compared to the number of steps
        """
        # Initialises the buffer of a given size
        exp_replay_size = int(num_steps * mem_pool_to_steps_ratio)
        self.mem_pool = NstepReplayMem(memory_size=exp_replay_size, n_steps=2)

    def setup_training_parameters(self, max_steps):
        """
        Initialises the training parameters
        :param max_steps: Max number of training steps
        """
        # Sets the learning rate and initial epsilon value
        self.learning_rate = self.hyperparams['learning_rate']
        self.eps_start = self.hyperparams['epsilon_start']
        # Sets the epsilon decay rate and final value - and burn in steps with no decay
        eps_step_denominator = self.hyperparams['eps_step_denominator'] if 'eps_step_denominator' in self.hyperparams else 2
        self.eps_step = max_steps / eps_step_denominator
        self.eps_end = 0.1
        self.burn_in = 50
        # Number of steps before synchronising the networks
        self.net_copy_interval = 50

    def finalize(self):
        pass

    def take_snapshot(self):
        """
        synchronising the target network with the main network
        """
        self.old_net.load_state_dict(self.net.state_dict())

    def make_actions(self, t, **kwargs):
        """
        Makes the actions based on exploration vs exploitation
        :param t: time step
        :param kwargs: Keyword arguments
        :return: Actions
        """
        # If greedy action then return greedy selection
        greedy = kwargs['greedy'] if 'greedy' in kwargs else True
        if greedy:
            return self.do_greedy_actions(t)
        else:
            # If t is divisible by 2 (should this not be eps_step_denominator?)
            if t % 2 == 0:
                # Decay epsilon
                self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
                                              * (self.eps_step - max(0., self.step)) / self.eps_step)
                # If random value less than epsilon then select exploration action
                if self.local_random.random() < self.eps:
                    # Selects and returns current and next exploratory actions
                    exploration_actions_t0, exploration_actions_t1 = self.environment.exploratory_actions(self.agent_exploration_policy)
                    self.next_exploration_actions = exploration_actions_t1
                    return exploration_actions_t0
                # Otherwise return the greedy actions
                else:
                    greedy_acts = self.do_greedy_actions(t)
                    self.next_exploration_actions = None
                    return greedy_acts
            # If not decay value then do not decay - return exploration if previous action was greedy? Seem odd
            else:
                if self.next_exploration_actions is not None:
                    return self.next_exploration_actions
                else:
                    return self.do_greedy_actions(t)

    def do_greedy_actions(self, time_t):
        # Get the current state
        cur_state = self.environment.get_state_ref()
        # Get the output actions
        actions, _, _ = self.net(time_t % 2, cur_state, None, greedy_acts=True)
        # Convert to list and return
        actions = list(actions.cpu().numpy())
        return actions

    def agent_exploration_policy(self, i):
        """
        :param i: Index of the graph under consideration
        :return: Returns a random action
        """
        return self.pick_random_actions(i)

    def run_simulation(self):
        """
        Runs a simulation step for the network
        """
        # Get the indices for the selected graphs
        selected_idx = self.advance_pos_and_sample_indices()
        self.environment.setup([self.train_g_list[idx] for idx in selected_idx],
                           [self.train_initial_obj_values[idx] for idx in selected_idx],
                           training=True)
        # This doesn't seem to do anything?
        self.post_env_setup()
        # Final state is none for each graph
        final_st = [None] * len(selected_idx)
        # Final acts filled to -1
        final_acts = np.empty(len(selected_idx), dtype=np.int)
        final_acts.fill(-1)
        # Set time equal to 0
        t = 0
        # While not a terminal state
        while not self.environment.is_terminal():
            # Get the greedy actions
            list_at = self.make_actions(t, greedy=False)
            # Find the environments where the budges has not been exhausted
            non_exhausted_before, = np.where(~self.environment.exhausted_budgets)
            # Convert to a list and take a step in the environment
            list_st = self.environment.clone_state(non_exhausted_before)
            self.environment.step(list_at)
            # Update the environments which have now got an exhausted action budget
            non_exhausted_after, = np.where(~self.environment.exhausted_budgets)
            exhausted_after, = np.where(self.environment.exhausted_budgets)
            # Environments with actions still available
            nonterm_indices = np.flatnonzero(np.isin(non_exhausted_before, non_exhausted_after))
            # Get the states which are non terminal before and after
            nonterm_st = [list_st[i] for i in nonterm_indices]
            nonterm_at = [list_at[i] for i in non_exhausted_after]
            # Set the rewards to zero
            rewards = np.zeros(len(nonterm_at), dtype=np.float)
            # Non terminal next states
            nonterm_s_prime = self.environment.clone_state(non_exhausted_after)
            # Get the indices of states that are now terminal
            now_term_indices = np.flatnonzero(np.isin(non_exhausted_before, exhausted_after))
            # Get the states
            term_st = [list_st[i] for i in now_term_indices]
            # Loop the states
            for i in range(len(term_st)):
                # Get the graph index
                g_list_index = non_exhausted_before[now_term_indices[i]]
                # Get the final states and actions
                final_st[g_list_index] = term_st[i]
                final_acts[g_list_index] = list_at[g_list_index]
            # If more than one state is non terminal then add to the memory pool
            if len(nonterm_at) > 0:
                self.mem_pool.add_list(nonterm_st, nonterm_at, rewards, nonterm_s_prime, [False] * len(nonterm_at), t % 2)
            # Increment the time step
            t += 1
        # List the final actions, next states, and get the environment rewards
        final_at = list(final_acts)
        rewards = self.environment.rewards
        final_s_prime = None
        # Add these final actions to the memory
        self.mem_pool.add_list(final_st, final_at, rewards, final_s_prime, [True] * len(final_at), (t - 1) % 2)

    def post_env_setup(self):
        """
        Doesn't appear to do anything!
        """
        pass

    def get_default_hyperparameters(self):
        """
        Sets the default training parameters
        :return: hyperparameters for training
        """
        hyperparams = {'learning_rate': 0.0001,
                       'epsilon_start': 1,
                       'mem_pool_to_steps_ratio': 1,
                       'latent_dim': 64,
                       'hidden': 32,
                       'embedding_method': 'mean_field',
                       'max_lv': 3,
                       'eps_step_denominator': 10}
        return hyperparams

