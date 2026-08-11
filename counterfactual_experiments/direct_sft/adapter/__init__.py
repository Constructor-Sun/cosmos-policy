from .latent_canonicalizer import (
    MODEL_NAMES,
    LatentCanonicalizer,
    build_model,
    load_model,
)

__all__ = [
    "MODEL_NAMES",
    "LatentCanonicalizer",
    "build_model",
    "load_model",
    "AdapterCosmosPolicyConfig",
    "AdapterCosmosPolicyVideo2WorldModel",
]


def __getattr__(name: str):
    if name == "AdapterCosmosPolicyConfig":
        from .model_config import AdapterCosmosPolicyConfig

        return AdapterCosmosPolicyConfig
    if name == "AdapterCosmosPolicyVideo2WorldModel":
        from .policy_model import AdapterCosmosPolicyVideo2WorldModel

        return AdapterCosmosPolicyVideo2WorldModel
    raise AttributeError(name)
