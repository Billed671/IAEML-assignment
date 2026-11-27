"""Neural networks used by the reinforcement-learning agents."""

import math

from flax import linen as nn


class ActorCritic(nn.Module):
    """Small MLP actor-critic for continuous actions.

    The actor predicts the mean of a Gaussian distribution in an unconstrained
    action space. A tanh transform is applied by the PPO implementation before
    actions are sent to the environment. The critic predicts the state value.
    """

    action_dim: int = 2
    hidden_size: int = 128

    @nn.compact
    def __call__(self, observations):
        actor = nn.Dense(
            self.hidden_size,
            kernel_init=nn.initializers.orthogonal(math.sqrt(2.0)),
        )(observations)
        actor = nn.tanh(actor)
        actor = nn.Dense(
            self.hidden_size,
            kernel_init=nn.initializers.orthogonal(math.sqrt(2.0)),
        )(actor)
        actor = nn.tanh(actor)

        critic = nn.Dense(
            self.hidden_size,
            kernel_init=nn.initializers.orthogonal(math.sqrt(2.0)),
        )(observations)
        critic = nn.tanh(critic)
        critic = nn.Dense(
            self.hidden_size,
            kernel_init=nn.initializers.orthogonal(math.sqrt(2.0)),
        )(critic)
        critic = nn.tanh(critic)

        action_mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )(actor)
        log_std = self.param(
            "log_std",
            nn.initializers.constant(-0.5),
            (self.action_dim,),
        )
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )(critic)
        return action_mean, log_std, value[..., 0]
