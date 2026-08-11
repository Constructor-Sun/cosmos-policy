"""Independent Cosmos Policy configuration entry point for adapter SFT."""

from cosmos_policy.config.config_v2 import ConfigV2, make_config_v2


Config = ConfigV2


def make_config():
    config = make_config_v2()
    from counterfactual_experiments.direct_sft.adapter.experiment import register_configs

    register_configs()
    return config


__all__ = ["Config", "make_config"]
