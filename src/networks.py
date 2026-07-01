import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, Optional


_ACTIVATION_FNS: dict[str, Callable] = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "silu": nn.silu,
    "tanh": jnp.tanh,
    "leaky_relu": nn.leaky_relu,
    "sigmoid": nn.sigmoid,
    "elu": nn.elu,
    "identity": lambda x: x,
}


def register_activation(name: str, fn: Callable) -> None:
    """Register a custom activation so it can be used by name in GeneralMLP."""
    _ACTIVATION_FNS[name] = fn


def get_activation(name: str) -> Callable:
    if name not in _ACTIVATION_FNS:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {sorted(_ACTIVATION_FNS)}"
        )
    return _ACTIVATION_FNS[name]


class Normalization(nn.Module):
    """Configurable normalization layer.

    Supported norm_type values: 'layer_norm', 'rms_norm'.
    """
    norm_type: str

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        match self.norm_type:
            case "layer_norm":
                return nn.LayerNorm()(x)
            case "rms_norm":
                return nn.RMSNorm()(x)
            case "none":
                return x
            case _:
                raise ValueError(
                    f"Unknown norm_type '{self.norm_type}'. "
                    "Available: 'layer_norm', 'rms_norm', 'none'"
                )


class GeneralMLP(nn.Module):
    """Feed-forward MLP with configurable depth, activation, and normalization.

    Args:
        hidden_dims:    Sequence of hidden layer widths.
        output_dim:     Width of the final linear output.
        normalization:  Normalization applied after each hidden linear layer.
                        Pass None to skip normalization, or a string key
                        accepted by Normalization ('layer_norm', 'rms_norm').
        activation:     Activation applied after normalization in each hidden
                        layer.  Any key registered in _ACTIVATION_FNS works
                        ('relu', 'gelu', 'silu', 'tanh', ...).  Use
                        register_activation() to add your own.
    """
    hidden_dims: tuple[int, ...]
    output_dim: int
    normalization: Optional[str] = None
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act_fn = get_activation(self.activation)
        for features in self.hidden_dims:
            x = nn.Dense(features)(x)
            if self.normalization is not None:
                x = Normalization(norm_type=self.normalization)(x)
            x = act_fn(x)
        return nn.Dense(self.output_dim)(x)
