import numpy as np

from rl.policy import Policy


class CustomPolicy(Policy):
    """Implement the Custom Q Policy

    This Custom Q Policy is a combination of Epsilon Greedy Policy and Boltzmann Q Policy with a parameter zeta
    """

    def __init__(self, zeta_start=1.0, zeta_delta=0.1, eps=.1, tau=1., clip=(-500., 500.)):
        super(CustomPolicy, self).__init__()
        self.zeta = zeta_start
        self.zeta_delta = zeta_delta
        self.eps = eps
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1

        def get_action_from_boltzmann(q, tau, clip):
            exp_values = np.exp(np.clip(q / tau, clip[0], clip[1]))
            probs = exp_values / np.sum(exp_values)
            a = np.random.choice(range(nb_actions), p=probs)
            return a

        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        if self.zeta < .5:
            action = get_action_from_boltzmann(q_values, self.tau, self.clip)
        else:
            if np.random.uniform() < self.eps:
                action = get_action_from_boltzmann(q_values, self.tau, self.clip)
            else:
                action = np.argmax(q_values)

        if self.zeta > 0:
            self.zeta -= self.zeta_delta

        return action

    def get_config(self):
        """Return configurations of CustomPolicy

        # Returns
            Dict of config
        """
        config = super(CustomPolicy, self).get_config()
        config['zeta'] = self.zeta
        config['zeta_delta'] = self.zeta_delta
        config['eps'] = self.eps
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config
