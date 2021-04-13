import gym
import numpy as np


class MC1DWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        self.high = obs_space.high[0]
        self.low = obs_space.low[0]
        self.observation_space = gym.spaces.Box(
            shape=(1,), low=0.0, high=self.high + abs(self.low)
        )

    def observation(self, obs):
        pos, vel = obs
        return np.expand_dims((pos + abs(self.low)) * np.sign(vel), 0)


def load_mc1d():
    import deep_control as dc

    env = dc.envs.load_gym("MountainCarContinuous-v0")
    return MC1DWrapper(env)


if __name__ == "__main__":
    import argparse
    import deep_control as dc
    import efr, efr2

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True)
    args = parser.parse_args()

    test_env = load_mc1d()
    state_space = test_env.observation_space
    action_space = test_env.action_space

    # create agent
    agent = efr2.EFRAgent(
        state_space.shape[0],
        action_space.shape[0],
        -10.0,
        2.0,
        2,
        advantage_method="mean",
        use_popart=True,
    )

    agent.load(args.agent)

    dc.run.evaluate_agent(agent, test_env, 20, 1000, True)
