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
@click.option(
    "--format",
    "fmt",
    default="onnx",
    help="Export format: onnx, torchscript, tensorrt, coreml, ros2",
)
@click.option("--output", default="./export/", help="Output directory")
@click.option("--fp16/--no-fp16", default=True, help="TensorRT FP16 precision (default: on)")
@click.option("--int8", is_flag=True, default=False, help="TensorRT INT8 quantization")
@click.option("--node-name", default="worldkit_node", help="ROS2 package/node name (ros2 only)")
def export(model, fmt, output, fp16, int8, node_name):
    """Export model for deployment."""
    from worldkit import WorldModel

    m = WorldModel.load(model)
    if fmt == "ros2":
        from worldkit.export.ros2_export import export_ros2

        path = export_ros2(m, output_dir=output, node_name=node_name)
    else:
        path = m.export(format=fmt, output=output, fp16=fp16, int8=int8)
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
@click.option("--model", required=True, help="Path to .wk model file")
@click.option("--data", required=True, help="Path to HDF5 file with observations")
@click.option("--labels", required=True, help="Path to labels file (CSV or HDF5)")
@click.option("--properties", required=True, help="Comma-separated property names to probe")
@click.option("--alpha", default=1.0, help="Ridge regularization strength")
@click.option("--test-fraction", default=0.2, help="Fraction held out for evaluation")
@click.option("--seed", default=42, help="Random seed")
def probe(model, data, labels, properties, alpha, test_fraction, seed):
    """Train linear probes on latent space to measure learned properties."""
    from worldkit import WorldModel

    m = WorldModel.load(model)
    prop_list = [p.strip() for p in properties.split(",")]
    result = m.probe(
        data=data,
        properties=prop_list,
        labels=labels,
        alpha=alpha,
        test_fraction=test_fraction,
        seed=seed,
    )
    click.echo(result.summary)


@cli.command()
@click.argument("path")
def validate(path):
    """Validate a .wk model file structure."""
    from worldkit.core.format import WKFormat

    try:
        WKFormat.validate(path)
        click.echo(f"Valid .wk archive: {path}")
    except ValueError as e:
        click.echo(f"Invalid: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.argument("path")
def inspect(path):
    """Inspect a .wk model file without loading weights."""
    from worldkit.core.format import WKFormat

    try:
        info = WKFormat.inspect(path)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    config = info["config"]
    meta = info["metadata"]
    act = info["action_space"]
    weights_mb = info["weights_size_bytes"] / (1024 * 1024)

    click.echo(f"Config:          {config.name}")
    click.echo(f"Latent dim:      {config.latent_dim}")
    click.echo(f"Image size:      {config.image_size}")
    click.echo(f"Action dim:      {config.action_dim}")
    click.echo(f"Weights size:    {weights_mb:.2f} MB")
    click.echo(f"Format version:  {meta.get('format_version', 'unknown')}")
    click.echo(f"WorldKit version:{meta.get('worldkit_version', 'unknown')}")
    click.echo(f"Created at:      {meta.get('created_at', 'unknown')}")
    if act:
        click.echo(f"Action type:     {act.get('type', 'unknown')}")
        click.echo(f"Action bounds:   [{act.get('low')}, {act.get('high')}]")


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
def env():
    """Discover and install environments."""
    pass


@env.command("list")
@click.option("--category", default=None, help="Filter by category")
def env_list(category):
    """List all registered environments."""
    from worldkit.envs.registry import registry

    envs = (
        registry.list_by_category(category) if category else registry.list_all()
    )
    if not envs:
        click.echo("No environments found.")
        return
    for e in envs:
        gym_tag = f"  gym={e.gym_id}" if e.gym_id else ""
        click.echo(
            f"  {e.env_id:<30} {e.category:<15} {e.action_type}{gym_tag}"
        )


@env.command("info")
@click.argument("env_id")
def env_info(env_id):
    """Show detailed information about an environment."""
    from worldkit.envs.registry import registry

    e = registry.get(env_id)
    click.echo(f"ID:           {e.env_id}")
    click.echo(f"Name:         {e.display_name}")
    click.echo(f"Category:     {e.category}")
    click.echo(f"Gym ID:       {e.gym_id or 'N/A'}")
    click.echo(f"Action dim:   {e.action_dim}")
    click.echo(f"Action type:  {e.action_type}")
    click.echo(f"Action range: [{e.action_low}, {e.action_high}]")
    click.echo(f"Obs shape:    {e.observation_shape}")
    click.echo(f"Description:  {e.description}")
    if e.install_cmd:
        click.echo(f"Install:      {e.install_cmd}")
    if e.dataset_url:
        click.echo(f"Dataset:      {e.dataset_url}")
    if e.pretrained_models:
        click.echo(f"Models:       {', '.join(e.pretrained_models)}")


@env.command("search")
@click.argument("query")
def env_search(query):
    """Search environments by keyword."""
    from worldkit.envs.registry import registry

    results = registry.search(query)
    if not results:
        click.echo(f"No environments matching '{query}'.")
        return
    for e in results:
        click.echo(f"  {e.env_id:<30} {e.description}")


@env.command("install")
@click.argument("env_id")
def env_install(env_id):
    """Install dependencies for an environment."""
    import subprocess
    import sys

    from worldkit.envs.registry import registry

    e = registry.get(env_id)
    if not e.install_cmd:
        click.echo(f"{env_id} has no additional dependencies to install.")
        return
    click.echo(f"Installing dependencies for {env_id}: {e.install_cmd}")
    subprocess.check_call([sys.executable, "-m", *e.install_cmd.split()[1:]])
    click.echo(f"Done. {env_id} is ready to use.")


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


@cli.group()
def bench():
    """WorldKit-Bench — evaluate world models."""
    pass


@bench.command("run")
@click.option("--model", required=True, help="Path to .wk model file")
@click.option(
    "--suite",
    default="full",
    help="Suite: full, quick, or a category (navigation, control, etc.)",
)
@click.option("--episodes", default=50, help="Episodes per task")
@click.option("--seed", default=42, help="Random seed")
@click.option("--output", default=None, help="Output JSON path for results")
def bench_run(model, suite, episodes, seed, output):
    """Run benchmark evaluation against a model."""
    from worldkit import WorldModel
    from worldkit.bench import BenchmarkRunner, BenchmarkSuite

    m = WorldModel.load(model)

    if suite == "full":
        s = BenchmarkSuite.full()
    elif suite == "quick":
        s = BenchmarkSuite.quick()
    else:
        s = BenchmarkSuite.category(suite)

    runner = BenchmarkRunner(s, m)
    results = runner.run(episodes_per_task=episodes, seed=seed)

    if output:
        results.to_json(output)
        click.echo(f"Results saved to {output}")


@bench.command("quick")
@click.option("--model", required=True, help="Path to .wk model file")
@click.option("--seed", default=42, help="Random seed")
def bench_quick(model, seed):
    """Run quick benchmark (5 tasks, 10 episodes each)."""
    from worldkit import WorldModel
    from worldkit.bench import BenchmarkRunner, BenchmarkSuite

    m = WorldModel.load(model)
    s = BenchmarkSuite.quick()
    runner = BenchmarkRunner(s, m)
    runner.run(episodes_per_task=10, seed=seed)


@bench.command("report")
@click.option("--results", "results_path", required=True, help="Path to results JSON")
@click.option("--output", required=True, help="Output HTML path")
def bench_report(results_path, output):
    """Generate HTML report from benchmark results."""
    from worldkit.bench import BenchmarkResults

    results = BenchmarkResults.from_json(results_path)
    results.to_html(output)
    click.echo(f"Report saved to {output}")


@cli.command()
@click.option(
    "--models", required=True, multiple=True, help="Paths to .wk model files"
)
@click.option("--data", required=True, help="Path to HDF5 evaluation data")
@click.option("--episodes", default=50, help="Number of episodes to evaluate")
@click.option(
    "--output", default="comparison.html", help="Output HTML report path"
)
def compare(models, data, episodes, output):
    """Compare multiple world models on the same dataset."""
    from pathlib import Path

    from worldkit import WorldModel
    from worldkit.eval.comparison import ModelComparator

    if len(models) < 2:
        click.echo("Error: provide at least 2 models to compare.", err=True)
        raise SystemExit(1)

    loaded: dict[str, WorldModel] = {}
    for model_path in models:
        name = Path(model_path).stem
        loaded[name] = WorldModel.load(model_path)

    comparator = ModelComparator(loaded)
    result = comparator.compare(data, episodes=episodes)
    report_path = comparator.report(result, save_to=output)

    click.echo(f"Best model: {result.best_model}")
    click.echo(f"Report saved to {report_path}")


@cli.group()
def federated():
    """Federated training across multiple clients."""
    pass


@federated.command("server")
@click.option("--model", required=True, help="Path to base .wk model file")
@click.option("--min-clients", default=2, help="Minimum clients before starting")
@click.option("--rounds", default=50, help="Number of federated rounds")
@click.option("--port", default=8080, help="Server port")
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--strategy", default="fedavg", help="Aggregation strategy")
@click.option("--output", default="./federated_model.wk", help="Output model path")
def federated_server(model, min_clients, rounds, port, host, strategy, output):
    """Start a federated training server."""
    import asyncio

    from worldkit import WorldModel
    from worldkit.federated import FederatedServer

    m = WorldModel.load(model)
    server = FederatedServer(m, min_clients=min_clients, strategy=strategy)
    asyncio.run(server.run(rounds=rounds, host=host, port=port))
    m.save(output)
    click.echo(f"Federated model saved to {output}")


@federated.command("client")
@click.option("--model", required=True, help="Path to base .wk model file")
@click.option("--data", required=True, help="Path to local HDF5 training data")
@click.option(
    "--server", "server_url", required=True,
    help="Server WebSocket URL (e.g. ws://localhost:8080/ws/federated)",
)
@click.option("--rounds", default=50, help="Number of federated rounds")
@click.option("--epochs", default=1, help="Local epochs per round")
@click.option("--batch-size", default=64, help="Local training batch size")
@click.option("--lr", default=1e-4, type=float, help="Local learning rate")
@click.option("--output", default=None, help="Save local model after training")
def federated_client(model, data, server_url, rounds, epochs, batch_size, lr, output):
    """Join a federated training session as a client."""
    import asyncio

    from worldkit import WorldModel
    from worldkit.federated import FederatedClient

    m = WorldModel.load(model)
    client = FederatedClient(m, server_url=server_url)
    asyncio.run(client.run(
        data=data,
        rounds=rounds,
        epochs_per_round=epochs,
        batch_size=batch_size,
        lr=lr,
    ))
    if output:
        m.save(output)
        click.echo(f"Local model saved to {output}")


if __name__ == "__main__":
    cli()
