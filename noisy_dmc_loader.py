import os
import numpy as np

import rl_utils as dc


def load(name, very_bad, bad, ok, good, expert, base_path=None):
    assert base_path is not None
    base_path = os.path.join(base_path, f"{name}_dset")
    sizes = [very_bad, bad, ok, good, expert]
    all_s, all_a, all_r, all_s1, all_d = [], [], [], [], []
    # load the data from each kind of demonstration
    for i, quality in enumerate(["very_bad", "bad", "ok", "good", "expert"]):
        filename = os.path.join(base_path, f"{quality}.npz")
        try:
            with open(filename, "rb") as f:
                arrays = np.load(f)
                # select a random subsample
                random_sample = np.random.choice(
                    arrays["states"].shape[0], sizes[i], replace=False
                )
                all_s.append(arrays["states"][random_sample])
                all_a.append(arrays["actions"][random_sample])
                all_r.append(arrays["rewards"][random_sample])
                all_s1.append(arrays["next_states"][random_sample])
                all_d.append(arrays["dones"][random_sample])
        except FileNotFoundError:
            print(f"Warning: no {quality} file detected...")
    # shuffle the data
    shuffle = np.random.permutation(sum(sizes))
    all_s = np.concatenate(all_s)[shuffle]
    all_a = np.concatenate(all_a)[shuffle]
    all_r = np.concatenate(all_r)[shuffle]
    all_s1 = np.concatenate(all_s1)[shuffle]
    all_d = np.concatenate(all_d)[shuffle]
    domain, task = name.split("_")
    env = dc.envs.load_dmc(domain, task)
    return env, (all_s, all_a, all_r, all_s1, all_d)


if __name__ == "__main__":
    env, (s, a, r, s1, d) = load("cheetah_run", 0, 0, 0, 0, 250_000)
    breakpoint()
