import numpy as np


# sample wrappers
def obs_wrapper(obs: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [obs['P'].flatten(), obs['Q'].flatten(), obs['Top'].flatten()])


def action_wrapper(action: int, bit_dim: int) -> np.ndarray:
    return np.array([int(action) >> i & 1 for i in range(bit_dim)][::-1])
