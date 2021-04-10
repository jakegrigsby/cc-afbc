from torch import nn
import torch.nn.functional as F
import torch
from deep_control import device
import math
from scipy import stats
import numpy as np


def get_linear_schedule(start, stop, steps):
    slope = (stop - start) / float(steps)
    upper = stop if slope >= 0.0 else start
    lower = start if slope >= 0.0 else stop

    def _sched(step):
        return max(min(start + (step * slope), upper), lower)

    return _sched


class AdvEstimatorFilter(nn.Module):
    def __init__(self, adv_estimator, filter_type="binary", beta=1.0):
        super().__init__()
        self.adv_estimator = adv_estimator
        self.filter_type = filter_type
        self.beta = beta
        self._norm_a2 = 0.5
        self._p_sched = get_linear_schedule(1.0, 0.05, 500_000)

    def get_hparams(self):
        return {"filter_type": self.filter_type, "filter_beta": self.beta}

    def forward(self, s, a, step_num=None):
        adv = self.adv_estimator(s, a)
        if self.filter_type == "exp":
            filter_val = (self.beta * adv).exp()
        elif self.filter_type == "exp_clamp":
            filter_val = (self.beta * adv.clamp(-5.0, 5.0)).exp()
        elif self.filter_type == "binary":
            filter_val = (adv >= 0.0).float()
        elif self.filter_type == "binary_t":
            adv_sample = torch.stack(
                [self.adv_estimator(s, a, n=4) for _ in range(10)], dim=1
            )
            with torch.no_grad():
                dist = self.adv_estimator.actor(s)
            adv_baseline = torch.stack(
                [self.adv_estimator(s, dist.sample(), n=4) for _ in range(10)], dim=1
            )
            t, p = stats.ttest_rel(
                adv_sample.cpu().numpy(),
                adv_baseline.cpu().numpy(),
                axis=1,
                alternative="greater",
            )
            p = torch.from_numpy(p).to(device)
            p_cutoff = self._p_sched(step_num) if step_num is not None else 0.05
            filter_val = (p <= p_cutoff).float()
        elif self.filter_type == "exp_norm":
            self._norm_a2 += 1e-5 * (adv.mean() ** 2 - self._norm_a2)
            norm_a = a / ((self._norm_a2).sqrt() + 1e-5)
            filter_val = (self.beta * norm_a).exp()
        elif self.filter_type == "softmax":
            batch_size = s.shape[0]
            filter_val = batch_size * F.softmax(self.beta * adv, dim=0)
        elif self.filter_type == "identity":
            filter_val = torch.ones_like(adv)
        else:
            raise ValueError(f"Unrecognized filter type '{self.filter_type}'")
        # final clip for numerical stability (only applies to exp filters)
        return filter_val.clamp(-100.0, 100.0)


class AdvClassifierFilter(nn.Module):
    def __init__(
        self,
        adv_estimator,
        model_t,
        model_kwargs,
        ensemble_size,
        max_unct_init=1.0,
        max_unct_final=0.05,
        max_unct_anneal=375_000,
        min_conf_init=0.01,
        min_conf_final=0.9,
        min_conf_anneal=375_000,
    ):
        super().__init__()
        self.adv_estimator = adv_estimator
        self.classifiers = [
            model_t(**model_kwargs).to(device) for _ in range(ensemble_size)
        ]

        self.unct_sched = get_linear_schedule(
            max_unct_init, max_unct_final, max_unct_anneal
        )
        self.conf_sched = get_linear_schedule(
            min_conf_init, min_conf_final, min_conf_anneal
        )

        self._hparams = {
            "filter_type": "classifier",
            "max_unct_init": max_unct_init,
            "max_unct_final": max_unct_final,
            "max_unct_anneal": max_unct_anneal,
            "min_conf_init": min_conf_init,
            "min_conf_final": min_conf_final,
            "min_conf_anneal": min_conf_anneal,
            "adv_classifier_ensemble_size": ensemble_size,
        }

    def get_hparams(self):
        return self._hparams

    def _generate_mask(self, state, action, step_num):
        adv_preds = torch.stack(
            [torch.sigmoid(a(state, action)) for a in self.classifiers], dim=0
        )
        mean_adv_preds = adv_preds.mean(0)
        min_conf = self.conf_sched(step_num)
        max_unct = self.unct_sched(step_num)
        mask = torch.zeros((state.shape[0], 1)).float().to(device)
        # high confidence positive examples get a +1 label
        mask[torch.where(mean_adv_preds >= min_conf)] = 1.0
        # any of those high confidence labels that are too
        # uncertain are set back to 0
        mask[torch.where(adv_preds.std(0) >= max_unct)] = 0.0
        return mask.squeeze(1)

    def generate_labels(self, states, actions):
        labels = torch.zeros((states.shape[0], 1))
        labels[torch.where(self.adv_estimator(states, actions) > 0.0)] = 1.0
        return labels.to(device)

    def forward(self, s, a, step_num):
        return self._generate_mask(s, a, step_num)
