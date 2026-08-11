"""Hydra experiment registration for isolated latent-adapter SFT."""

from hydra.core.config_store import ConfigStore

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict

from .policy_model import AdapterCosmosPolicyVideo2WorldModel


EXPERIMENT_NAME = "cosmos_predict2_2b_480p_libero_adapter"

adapter_libero_experiment = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(AdapterCosmosPolicyVideo2WorldModel)(
            adapter_config=dict(
                adapter_model_name="transformer",
                adapter_channels=16,
                adapter_slots=2,
                adapter_height=28,
                adapter_width=28,
                adapter_hidden_dim=64,
                adapter_mlp_dim=128,
                adapter_num_heads=4,
                adapter_dropout=0.0,
                adapter_gate_hidden_dim=32,
                adapter_initial_gate=0.12,
                adapter_freeze_backbone=True,
            )
        ),
        job=dict(
            group="counterfactual_adapter_sft",
            name=EXPERIMENT_NAME,
        ),
    )
)


def register_configs() -> None:
    ConfigStore.instance().store(
        group="experiment",
        package="_global_",
        name=EXPERIMENT_NAME,
        node=adapter_libero_experiment,
    )
