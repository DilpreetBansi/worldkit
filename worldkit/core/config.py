"""Model configurations for WorldKit."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a WorldKit world model."""

    name: str = "base"

    # Encoder
    encoder_name: str = "google/vit-base-patch16-224"
    encoder_embed_dim: int = 768
    image_size: int = 96
    patch_size: int = 16

    # Latent space
    latent_dim: int = 192
    proj_hidden_dim: int = 384
    proj_output_dim: int = 192

    # Predictor
    pred_depth: int = 3
    pred_heads: int = 4
    pred_dim_head: int = 64
    pred_mlp_dim: int = 384
    pred_dropout: float = 0.0
    pred_emb_dropout: float = 0.0
    context_length: int = 3

    # Action encoder
    action_dim: int = 2
    action_embed_dim: int = 192

    # SIGReg
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024

    # Training
    lambda_reg: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    batch_size: int = 64
    num_workers: int = 4
    sequence_length: int = 16


CONFIGS = {}


def register_config(name: str, **overrides):
    """Register a named model configuration."""
    config = ModelConfig(name=name, **overrides)
    CONFIGS[name] = config
    return config


register_config(
    "nano",
    encoder_name="google/vit-tiny-patch16-224",
    encoder_embed_dim=192,
    latent_dim=128,
    proj_hidden_dim=256,
    proj_output_dim=128,
    pred_depth=2,
    pred_heads=4,
    pred_dim_head=32,
    pred_mlp_dim=256,
    action_embed_dim=128,
)

register_config(
    "base",
    encoder_name="google/vit-small-patch16-224",
    encoder_embed_dim=384,
    latent_dim=192,
    proj_hidden_dim=384,
    proj_output_dim=192,
    pred_depth=3,
    pred_heads=4,
    pred_dim_head=64,
    pred_mlp_dim=384,
    action_embed_dim=192,
)

register_config(
    "large",
    encoder_name="google/vit-base-patch16-224",
    encoder_embed_dim=768,
    latent_dim=384,
    proj_hidden_dim=768,
    proj_output_dim=384,
    pred_depth=4,
    pred_heads=8,
    pred_dim_head=64,
    pred_mlp_dim=768,
    action_embed_dim=384,
)

register_config(
    "xl",
    encoder_name="google/vit-large-patch16-224",
    encoder_embed_dim=1024,
    latent_dim=512,
    proj_hidden_dim=1024,
    proj_output_dim=512,
    pred_depth=6,
    pred_heads=8,
    pred_dim_head=64,
    pred_mlp_dim=1024,
    action_embed_dim=512,
)


def get_config(name: str = "base", **overrides) -> ModelConfig:
    """Get a model configuration by name, with optional overrides."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    import dataclasses

    config = dataclasses.replace(CONFIGS[name], **overrides)
    return config
