"""Resume SAC with policy-driven collection into an empty replay buffer."""

from stable_baselines3 import SAC


class PolicyWarmupSAC(SAC):
    """Separate the update warm-up from uniform random action selection.

    SB3 normally uses ``learning_starts`` for both. Here it still delays
    gradient updates, but actions always come from the loaded stochastic
    policy. The parent sampler retains action scaling and replay storage
    conventions. Use this class only when resuming without a replay buffer;
    fresh training and resume with a restored buffer keep standard SAC.
    """

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # Override only the sampler's threshold, not self.learning_starts:
        # learn() must still wait for enough newly collected experience.
        return super()._sample_action(
            learning_starts=0,
            action_noise=action_noise,
            n_envs=n_envs,
        )
