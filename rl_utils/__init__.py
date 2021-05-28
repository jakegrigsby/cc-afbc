import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from . import (
    sac,
    td3,
    sbc,
)
