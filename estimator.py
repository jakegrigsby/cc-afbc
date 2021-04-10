from torch import nn
import torch.nn.functional as F
import torch
import random


class AdvantageEstimator(nn.Module):
    def __init__(self, actor, critics, popart=False, method="mean", ensembling="min"):
        super().__init__()
        assert method in ["mean", "max", "expectation"]
        assert ensembling in ["min", "mean"]
        self.actor = actor
        self.critics = critics
        self.method = method
        self.ensembling = ensembling
        self.val_s = None
        self.popart = popart

    def pop(self, q, s, a):
        if self.popart:
            return self.popart(q(s, a))
        else:
            return q(s, a)

    def get_hparams(self):
        return {"adv_method": self.method, "adv_ensembling_method": self.method}

    def estimate_value(self, state, n=10):
        # get an action distribution from the policy
        act_dist = self.actor(state)
        actions = [act_dist.sample() for _ in range(n)]

        # get the q value for each of the n actions
        qs = []
        for act in actions:
            q_preds = torch.stack(
                [self.pop(critic, state, act) for critic in self.critics], dim=0
            )
            if self.ensembling == "min":
                q_preds = q_preds.min(0).values
            elif self.ensembling == "mean":
                q_preds = q_preds.mean(0)
            qs.append(q_preds)

        if self.method == "mean":
            # V(s) = E_{a ~ \pi(s)} [Q(s, a)]
            value = torch.stack(qs, dim=0).mean(0)
        elif self.method == "max":
            # Optimisitc value estimate: V(s) = max_{a1, a2, a3, ..., aN}(Q(s, a))
            value = torch.stack(qs, dim=0).max(0).values
        elif self.method == "expectation":
            probs = F.softmax(
                torch.stack(
                    [
                        act_dist.log_prob(act).sum(-1, keepdim=True).exp()
                        for act in actions
                    ],
                    dim=0,
                ),
                dim=0,
            )
            value = (probs * torch.stack(qs, dim=0)).sum(0)

        self.val_s = value
        return value

    def stochastic_qval(self, s, a, n=10):
        with torch.no_grad():
            qs = [random.choice(self.critics)(s, a) for _ in range(n)]
        return torch.stack(qs, dim=1)

    def stochastic_val(self, s, a, n=10):
        with torch.no_grad():
            dist = self.actor(s)
        vs = [random.choice(self.critics)(s, dist.sample()) for _ in range(n)]
        return torch.stack(vs, dim=1)

    def forward(self, state, action, use_computed_val=False, n=10):
        with torch.no_grad():
            q_preds = torch.stack(
                [self.pop(critic, state, action) for critic in self.critics], dim=0
            )
            if self.ensembling == "min":
                q_preds = q_preds.min(0).values
            elif self.ensembling == "mean":
                q_preds = q_preds.mean(0)
            # reuse the expensive value computation if it has already been done
            if use_computed_val:
                assert self.val_s is not None
            else:
                # do the value computation
                self.estimate_value(state, n=n)
            # A(s, a) = Q(s, a) - V(s)
            adv = q_preds - self.val_s
        return adv
