"""WorldKit CLI.

Usage:
    worldkit train --data ./data.h5 --config base --epochs 100
    worldkit serve --model ./model.wk --port 8000
    worldkit export --model ./model.wk --format onnx
    worldkit hub list
    worldkit info ./model.wk
"""

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="worldkit")
def cli():
    """WorldKit — The open-source world model runtime."""
    pass


@cli.command()
@click.option("--data", required=True, help="Path to HDF5 training data")
@click.option("--config", default="base", help="Model config: nano, base, large, xl")
@click.option("--epochs", default=100, help="Training epochs")
@click.option("--batch-size", default=64, help="Batch size")
@click.option("--lr", default=1e-4, help="Learning rate")
@click.option("--lambda-reg", default=1.0, help="SIGReg weight (the ONE hyperparameter)")
@click.option("--action-dim", default=2, help="Action space dimensionality")
@click.option("--device", default="auto", help="Device: cuda, cpu, mps, auto")
@click.option("--output", default="./model.wk", help="Output model path")
@click.option("--seed", default=42, help="Random seed")
def train(data, config, epochs, batch_size, lr, lambda_reg, action_dim, device, output, seed):
    """Train a world model."""
    from worldkit import WorldModel

    model = WorldModel.train(
        data=data,
        config=config,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_reg=lambda_reg,
        action_dim=action_dim,
        device=device,
        seed=seed,
    )
    model.save(output)
    click.echo(f"Model saved to {output}")


@cli.command()
@click.option("--model", required=True, help="Path to .wk model file")
@click.option("--format", "fmt", default="onnx", help="Export format: onnx, torchscript")
@click.option("--output", default="./export/", help="Output directory")
def export(model, fmt, output):
    """Export model for deployment."""
    from worldkit import WorldModel

    m = WorldModel.load(model)
    path = m.export(format=fmt, output=output)
    click.echo(f"Exported to {path}")


@cli.command()
@click.option("--model", required=True, help="Path to .wk model file")
@click.option("--port", default=8000, help="Server port")
@click.option("--host", default="0.0.0.0", help="Server host")
def serve(model, port, host):
    """Start inference API server."""
    import os

    import uvicorn

    os.environ["WORLDKIT_MODEL_PATH"] = model
    uvicorn.run("worldkit.server.app:app", host=host, port=port, reload=False)


@cli.command()
@click.option("--model", required=True, help="Path to .wk model file")
def info(model):
    """Show model information."""
    from worldkit import WorldModel

    m = WorldModel.load(model)
    click.echo(f"Config:     {m.config.name}")
    click.echo(f"Parameters: {m.num_params:,}")
    click.echo(f"Latent dim: {m.latent_dim}")
    click.echo(f"Device:     {m.device}")


@cli.command()
@click.option("--input", "input_dir", required=True, help="Input video directory")
@click.option("--output", required=True, help="Output HDF5 path")
@click.option("--fps", default=10, help="Target FPS")
def convert(input_dir, output, fps):
    """Convert videos to WorldKit HDF5 format."""
    from worldkit.data import Converter

    c = Converter()
    c.from_video(input_dir=input_dir, output=output, fps=fps)


@cli.command()
@click.option("--env", required=True, help="Gymnasium environment ID")
@click.option("--episodes", default=100, help="Number of episodes")
@click.option("--output", required=True, help="Output HDF5 path")
@click.option("--max-steps", default=500, help="Max steps per episode")
def record(env, episodes, output, max_steps):
    """Record environment interactions to HDF5."""
    import gymnasium as gym

    from worldkit.data import Recorder

    environment = gym.make(env, render_mode="rgb_array")
    rec = Recorder(environment, output=output)
    rec.record(episodes=episodes, max_steps_per_episode=max_steps)


@cli.group()
def hub():
    """Interact with WorldKit Hub."""
    pass


@hub.command("list")
def hub_list():
    """List available pre-trained models."""
    from huggingface_hub import HfApi

    api = HfApi()
    models = api.list_models(author="worldkit", sort="downloads", direction=-1)
    for model in models:
        click.echo(f"  {model.modelId:<40} downloads: {model.downloads}")


@hub.command("download")
@click.argument("model_id")
@click.option("--output", default=".", help="Download directory")
def hub_download(model_id, output):
    """Download a model from the hub."""
    from worldkit import WorldModel

    model = WorldModel.from_hub(model_id)
    out_path = f"{output}/{model_id.split('/')[-1]}.wk"
    model.save(out_path)
    click.echo(f"Downloaded to {out_path}")


if __name__ == "__main__":
    cli()
