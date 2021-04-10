import os

import torch
from deep_control import nets, device
from popart import PopArtLayer


class AFBCAgent:
    def __init__(
        self,
        obs_space_size,
        act_space_size,
        log_std_low=-10.0,
        log_std_high=2.0,
        critics=2,
        actor_network_cls=nets.StochasticActor,
        critic_network_cls=nets.BigCritic,
        hidden_size=256,
        beta_dist=False,
        use_popart=False,
    ):
        self.actor = actor_network_cls(
            obs_space_size,
            act_space_size,
            log_std_low,
            log_std_high,
            dist_impl="beta" if beta_dist else "pyd",
            hidden_size=hidden_size,
        )
        self.critics = [
            critic_network_cls(obs_space_size, act_space_size, hidden_size)
            for _ in range(critics)
        ]
        if use_popart:
            self.popart = PopArtLayer()
        else:
            self.popart = False

        self._beta_dist = beta_dist
        self._log_std_low = log_std_low
        self._log_std_high = log_std_high
        self._hidden_size = hidden_size

    def get_hparams(self):
        hparams = {
            "log_std_low": self._log_std_low,
            "log_std_high": self._log_std_high,
            "critic_ensemble_size": len(self.critics),
            "popart": self.popart is not False,
            "beta_dist": self._beta_dist,
            "hidden_size": self._hidden_size,
        }
        return hparams

    def to(self, device):
        self.actor = self.actor.to(device)
        if self.popart:
            self.popart = self.popart.to(device)
        for i, critic in enumerate(self.critics):
            self.critics[i] = critic.to(device)

    def eval(self):
        self.actor.eval()
        if self.popart:
            self.popart.eval()
        for critic in self.critics:
            critic.eval()

    def train(self):
        self.actor.train()
        if self.popart:
            self.popart.train()
        for critic in self.critics:
            critic.train()

    def save(self, path):
        actor_path = os.path.join(path, "actor.pt")
        torch.save(self.actor.state_dict(), actor_path)
        if self.popart:
            popart_path = os.path.join(path, "popart.pt")
            torch.save(self.popart.state_dict(), popart_path)
        for i, critic in enumerate(self.critics):
            critic_path = os.path.join(path, f"critic{i}.pt")
            torch.save(critic.state_dict(), critic_path)

    def load(self, path):
        actor_path = os.path.join(path, "actor.pt")
        self.actor.load_state_dict(torch.load(actor_path))
        if self.popart:
            popart_path = os.path.join(path, "popart.pt")
            self.popart.load_state_dict(torch.load(popart_path))
        for i, critic in enumerate(self.critics):
            critic_path = os.path.join(path, f"critic{i}.pt")
            critic.load_state_dict(torch.load(critic_path))

    def forward(self, state, from_cpu=True):
        if from_cpu:
            state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = act_dist.mean
        self.actor.train()
        if from_cpu:
            act = self.process_act(act)
        return act

    def sample_action(self, state, from_cpu=True):
        if from_cpu:
            state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = act_dist.sample()
        self.actor.train()
        if from_cpu:
            act = self.process_act(act)
        return act

    def process_state(self, state):
        return torch.from_numpy(state).unsqueeze(0).float().to(device)

    def process_act(self, act):
        return act.clamp(-1.0, 1.0).squeeze(0).cpu().numpy()
