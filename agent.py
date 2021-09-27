import numpy as np
from model import Model
from utils import *

class PGAgent:
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']

        # actor/policy # This is the actor to predict the q values
        self.actor = Model(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

        ########################### Main Part ########################################

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        # Inputs are sampled with according to given batch_size

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # step 1: calculate q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values = self.calculate_q_vals(rewards_list)

        # step 2: calculate advantages that correspond to each (s_t, a_t) point
        advantages = self.estimate_advantage(observations, q_values)

        ################ Model update ###################
        train_log = self.actor.update(
            observations,
            actions,
            advantages,
            q_values,
        )

        return train_log

    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """
        """
                   Monte Carlo estimation of the Q function.
                   arguments:
                       rews_list: length: number of sampled rollouts
                           Each element corresponds to a particular rollout,
                           and contains an array of the rewards for every step of that particular rollout
                   returns:
                       q_values: shape: (sum/total number of steps across the rollouts)
                           Each entry corresponds to the estimated q(s_t,a_t) value 
                           of the corresponding obs/ac point at time t.

        """

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        if not self.reward_to_go:

            # For each point (s_t, a_t), associate its value as being the discounted sum of rewards over the full trajectory
            # In other words: value of (s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            q_values = np.concatenate([self._discounted_return(r) for r in rewards_list])

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:

            # For each point (s_t, a_t), associate its value as being the discounted sum of rewards over the full trajectory
            # In other words: value of (s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            q_values = np.concatenate([self._discounted_cumsum(r) for r in rewards_list])

        return q_values

    def estimate_advantage(self, obs, q_values): # This is the method to reduce the variance

        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the baseline
        if self.nn_baseline:
            baselines_unnormalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the baseline and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert baselines_unnormalized.ndim == q_values.ndim
            ## baseline was trained with standardized q_values, so ensure that the predictions
            ## have the same mean and standard deviation as the current batch of q_values
            baselines = baselines_unnormalized * np.std(q_values) + np.mean(q_values)
            advantages = q_values - baselines

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            ## and a standard deviation of one
            ## HINT: there is a `normalize` function in `infrastructure.utils`
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths): # Add recent observed paths to buffer
        self.replay_buffer.add_rollouts(paths)

    ##################### Agent Sample ###########################################
    def sample(self, batch_size): # This is for sampling data to train from replay buffer
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        # 1) create a list of indices (t'): from 0 to T-1
        indices = np.arange(0, len(rewards))

        # 2) create a list where the entry at each index (t') is gamma^(t')
        discounts = np.power(self.gamma, indices)

        # 3) create a list where the entry at each index (t') is gamma^(t') * r_{t'}
        discounted_rewards = discounts * rewards

        # 4) calculate a scalar: sum_{t'=0}^{T-1} gamma^(t') * r_{t'}
        sum_of_discounted_rewards = np.sum(discounted_rewards)

        # 5) create a list of length T-1, where each entry t contains that scalar
        list_of_discounted_returns = np.repeat(sum_of_discounted_rewards, len(rewards))

        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """
        all_discounted_cumsums = [] # create empty list

        for start_time_index in range(len(rewards)):
            # 1) create a list of indices (t'): goes from t to T-1
            indices = np.arange(start_time_index, len(rewards))

            # 2) create a list where the entry at each index (t') is gamma^(t'-t)
            discounts = np.power(self.gamma, indices - start_time_index) # discount vector

            # 3) create a list where the entry at each index (t') is gamma^(t'-t) * r_{t'}
            # Hint: remember that t' goes from t to T-1, so you should use the rewards from those indices as well
            discounted_rtg = discounts * rewards[start_time_index:] # discount the rewards

            # 4) calculate a scalar: sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}
            sum_discounted_rtg = np.sum(discounted_rtg) # sum discounted rewards

            # appending each of these calculated sums into the list to return
            all_discounted_cumsums.append(sum_discounted_rtg)

        list_of_discounted_cumsums = np.array(all_discounted_cumsums) # convert list to array

        return list_of_discounted_cumsums

class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size
        self.paths = []
        self.obs = None
        self.acs = None
        self.concatenated_rews = None
        self.unconcatenated_rews = None
        self.next_obs = None
        self.terminals = None

    def add_rollouts(self, paths, noised=False):

        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto our arrays
        observations, actions, next_observations, terminals, concatenated_rews, unconcatenated_rews = convert_listofrollouts(paths)

        if noised:
            observations = add_noise(observations)
            next_observations = add_noise(next_observations)

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
            self.concatenated_rews = concatenated_rews[-self.max_size:]
            self.unconcatenated_rews = unconcatenated_rews[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]
            self.concatenated_rews = np.concatenate(
                [self.concatenated_rews, concatenated_rews]
            )[-self.max_size:]
            if isinstance(unconcatenated_rews, list):
                self.unconcatenated_rews += unconcatenated_rews
            else:
                self.unconcatenated_rews.append(unconcatenated_rews)

    ########################################
    ########################################

    def sample_random_rollouts(self, num_rollouts):
        rand_indices = np.random.permutation(len(self.paths))[:num_rollouts]
        return self.paths[rand_indices]

    def sample_recent_rollouts(self, num_rollouts=1):
        return self.paths[-num_rollouts:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):

        assert self.obs.shape[0] == self.acs.shape[0] == self.concatenated_rews.shape[0] == self.next_obs.shape[0] == self.terminals.shape[0]
        rand_indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        return self.obs[rand_indices], self.acs[rand_indices], self.concatenated_rews[rand_indices], self.next_obs[rand_indices], self.terminals[rand_indices]

    def sample_recent_data(self, batch_size=1, concat_rew=True):

        if concat_rew:
            return self.obs[-batch_size:], self.acs[-batch_size:], self.concatenated_rews[-batch_size:], self.next_obs[-batch_size:], self.terminals[-batch_size:]
        else:
            num_recent_rollouts_to_return = 0
            num_datapoints_so_far = 0
            index = -1
            while num_datapoints_so_far < batch_size:
                recent_rollout = self.paths[index]
                index -=1
                num_recent_rollouts_to_return +=1
                num_datapoints_so_far += get_pathlength(recent_rollout)
            rollouts_to_return = self.paths[-num_recent_rollouts_to_return:]
            observations, actions, next_observations, terminals, concatenated_rews, unconcatenated_rews = convert_listofrollouts(rollouts_to_return)
            return observations, actions, unconcatenated_rews, next_observations, terminals