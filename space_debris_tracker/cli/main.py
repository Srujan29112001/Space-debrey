"""
Space Debris Tracker CLI
Command-line interface for space debris tracking operations
"""

import click
import sys
import logging
from pathlib import Path
from typing import Optional
import yaml
from datetime import datetime
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Quiet mode')
@click.pass_context
def cli(ctx, config, verbose, quiet):
    """
    Space Debris Tracking & Collision Prediction System

    A comprehensive AI-powered system for tracking space debris and
    predicting collision risks using computer vision, physics-informed
    neural networks, and autonomous monitoring agents.
    """
    ctx.ensure_object(dict)

    # Load configuration
    if config:
        with open(config, 'r') as f:
            ctx.obj['config'] = yaml.safe_load(f)
    else:
        # Default config
        default_config_path = Path(__file__).parent.parent.parent / 'config.yaml'
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                ctx.obj['config'] = yaml.safe_load(f)
        else:
            ctx.obj['config'] = {}

    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.WARNING)


@cli.group()
def data():
    """Data ingestion and processing commands"""
    pass


@data.command()
@click.option('--source', type=click.Choice(['space-track', 'celestrak', 'file']),
              required=True, help='TLE data source')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output file path')
@click.option('--catalog', type=str, help='Satellite catalog (e.g., active, debris)')
@click.option('--norad-id', type=int, help='Specific NORAD ID to fetch')
@click.option('--username', type=str, envvar='SPACETRACK_USERNAME',
              help='Space-Track.org username')
@click.option('--password', type=str, envvar='SPACETRACK_PASSWORD',
              help='Space-Track.org password')
@click.pass_context
def fetch_tle(ctx, source, output, catalog, norad_id, username, password):
    """Fetch TLE data from various sources"""
    from space_debris_tracker.data_ingestion import TLEParser

    click.echo(f"Fetching TLE data from {source}...")

    parser = TLEParser(
        space_track_username=username,
        space_track_password=password
    )

    try:
        if norad_id:
            # Fetch specific satellite
            elements = parser.fetch_latest_tle(norad_id)
            if elements:
                # Save to file
                with open(output, 'w') as f:
                    f.write(f"{elements.name}\n")
                    # Write TLE lines (would need to reconstruct from elements)
                click.echo(f"✓ Fetched TLE for {elements.name} (NORAD {norad_id})")
            else:
                click.echo(f"✗ Failed to fetch TLE for NORAD {norad_id}", err=True)
                sys.exit(1)
        else:
            # Fetch catalog
            elements_list = parser.fetch_catalog(
                classification='U',
                output_file=output
            )
            click.echo(f"✓ Fetched {len(elements_list)} TLEs to {output}")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@data.command()
@click.argument('tle_file', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['json', 'csv', 'yaml']),
              default='json', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file')
def parse_tle(tle_file, format, output):
    """Parse and validate TLE file"""
    from space_debris_tracker.data_ingestion import TLEParser
    import json
    import csv

    parser = TLEParser()

    try:
        elements_list = parser.parse_tle_file(tle_file)

        click.echo(f"Parsed {len(elements_list)} TLE sets")

        # Convert to desired format
        data = [elem.to_dict() for elem in elements_list]

        if output:
            if format == 'json':
                with open(output, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif format == 'csv':
                if data:
                    with open(output, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
            elif format == 'yaml':
                with open(output, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)

            click.echo(f"✓ Saved to {output}")
        else:
            # Print to stdout
            if format == 'json':
                click.echo(json.dumps(data, indent=2, default=str))
            elif format == 'yaml':
                click.echo(yaml.dump(data, default_flow_style=False))

    except Exception as e:
        click.echo(f"✗ Error parsing TLE: {e}", err=True)
        sys.exit(1)


@cli.group()
def train():
    """Model training commands"""
    pass


@train.command()
@click.option('--data-path', required=True, type=click.Path(exists=True),
              help='Path to training data')
@click.option('--model-type', type=click.Choice(['yolo', 'pinn', 'transformer', 'mamba']),
              required=True, help='Model type to train')
@click.option('--epochs', default=100, help='Number of training epochs')
@click.option('--batch-size', default=32, help='Batch size')
@click.option('--lr', default=0.001, help='Learning rate')
@click.option('--output-dir', default='checkpoints/', help='Output directory for checkpoints')
@click.option('--resume', type=click.Path(exists=True), help='Resume from checkpoint')
@click.option('--gpu', is_flag=True, help='Use GPU acceleration')
@click.pass_context
def start(ctx, data_path, model_type, epochs, batch_size, lr, output_dir, resume, gpu):
    """Start model training"""
    import torch
    from space_debris_tracker.training import create_dataloaders

    click.echo(f"Starting {model_type.upper()} training...")
    click.echo(f"  Data: {data_path}")
    click.echo(f"  Epochs: {epochs}")
    click.echo(f"  Batch size: {batch_size}")
    click.echo(f"  Learning rate: {lr}")
    click.echo(f"  Device: {'GPU' if gpu and torch.cuda.is_available() else 'CPU'}")

    device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'

    try:
        # Create dataloaders
        if model_type == 'yolo':
            from space_debris_tracker.training import YOLOv7Trainer
            from space_debris_tracker.computer_vision import SpaceDebrisDetector

            train_loader, val_loader, _ = create_dataloaders(
                dataset_type='detection',
                data_path=data_path,
                batch_size=batch_size
            )

            model = SpaceDebrisDetector()
            trainer = YOLOv7Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config={
                    'epochs': epochs,
                    'learning_rate': lr,
                    'device': device,
                    'checkpoint_dir': output_dir
                }
            )

        elif model_type == 'pinn':
            from space_debris_tracker.training import PINNTrainer
            from space_debris_tracker.trajectory_prediction.pinn import PhysicsInformedNN

            train_loader, val_loader, _ = create_dataloaders(
                dataset_type='orbit',
                data_path=data_path,
                batch_size=batch_size
            )

            model = PhysicsInformedNN()
            trainer = PINNTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config={
                    'epochs': epochs,
                    'learning_rate': lr,
                    'device': device,
                    'checkpoint_dir': output_dir
                }
            )

        elif model_type == 'transformer':
            from space_debris_tracker.training import TransformerTrainer
            from space_debris_tracker.trajectory_prediction.transformer import TrajectoryTransformer

            train_loader, val_loader, _ = create_dataloaders(
                dataset_type='orbit',
                data_path=data_path,
                batch_size=batch_size
            )

            model = TrajectoryTransformer()
            trainer = TransformerTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config={
                    'epochs': epochs,
                    'learning_rate': lr,
                    'device': device,
                    'checkpoint_dir': output_dir
                }
            )

        elif model_type == 'mamba':
            from space_debris_tracker.training import MemoryEfficientMambaTrainer
            from space_debris_tracker.trajectory_prediction.mamba import Mamba2Predictor

            train_loader, val_loader, _ = create_dataloaders(
                dataset_type='orbit',
                data_path=data_path,
                batch_size=batch_size
            )

            model = Mamba2Predictor()
            trainer = MemoryEfficientMambaTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config={
                    'epochs': epochs,
                    'learning_rate': lr,
                    'device': device,
                    'checkpoint_dir': output_dir
                }
            )

        # Resume from checkpoint if specified
        if resume:
            click.echo(f"Resuming from {resume}")
            trainer.load_checkpoint(resume)

        # Train
        click.echo("Training started...")
        trainer.train()

        click.echo(f"✓ Training completed! Checkpoints saved to {output_dir}")

    except Exception as e:
        click.echo(f"✗ Training error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.group()
def predict():
    """Prediction and analysis commands"""
    pass


@predict.command()
@click.argument('tle_file', type=click.Path(exists=True))
@click.option('--horizon', default=7, help='Prediction horizon in days')
@click.option('--model', type=click.Path(exists=True), help='Model checkpoint')
@click.option('--output', '-o', type=click.Path(), help='Output file')
@click.option('--format', type=click.Choice(['json', 'csv']), default='json')
def trajectory(tle_file, horizon, model, output, format):
    """Predict orbital trajectories"""
    from space_debris_tracker.data_ingestion import TLEParser
    from space_debris_tracker.trajectory_prediction import OrbitPredictionEngine
    import json

    parser = TLEParser()
    predictor = OrbitPredictionEngine()

    # Load model if provided
    if model:
        predictor.load_checkpoint(model)

    try:
        elements_list = parser.parse_tle_file(tle_file)

        click.echo(f"Predicting trajectories for {len(elements_list)} objects...")
        click.echo(f"Time horizon: {horizon} days")

        results = []

        with click.progressbar(elements_list, label='Processing') as bar:
            for elements in bar:
                # Get current state
                current_time = datetime.utcnow()
                pos, vel = parser.propagate_sgp4(elements, current_time)

                initial_state = {
                    'position': pos,
                    'velocity': vel,
                    'norad_id': elements.norad_id,
                    'name': elements.name
                }

                # Predict
                prediction = predictor.predict_trajectory(
                    initial_state=torch.tensor([*pos, *vel], dtype=torch.float32),
                    time_horizon=horizon * 86400,
                    dt=3600  # 1-hour steps
                )

                results.append({
                    'norad_id': elements.norad_id,
                    'name': elements.name,
                    'prediction_epoch': current_time.isoformat(),
                    'horizon_days': horizon,
                    'trajectory': prediction['trajectory'].tolist(),
                    'uncertainty': prediction['uncertainty'].tolist()
                })

        # Save results
        if output:
            with open(output, 'w') as f:
                if format == 'json':
                    json.dump(results, f, indent=2, default=str)
                elif format == 'csv':
                    # Flatten for CSV
                    import csv
                    with open(output, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['norad_id', 'name', 'prediction_epoch', 'horizon_days'])
                        for r in results:
                            writer.writerow([r['norad_id'], r['name'],
                                           r['prediction_epoch'], r['horizon_days']])

            click.echo(f"✓ Results saved to {output}")
        else:
            click.echo(json.dumps(results, indent=2, default=str))

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@predict.command()
@click.option('--satellite', required=True, type=int, help='Primary satellite NORAD ID')
@click.option('--threshold', default=0.0001, help='Minimum collision probability threshold')
@click.option('--days', default=7, help='Time window in days')
@click.option('--output', '-o', type=click.Path(), help='Output file')
def conjunctions(satellite, threshold, days, output):
    """Find conjunction events for a satellite"""
    from space_debris_tracker.knowledge_graph import SpaceKnowledgeGraph
    import json

    kg = SpaceKnowledgeGraph()

    click.echo(f"Finding conjunctions for satellite {satellite}...")
    click.echo(f"  Threshold: {threshold}")
    click.echo(f"  Time window: {days} days")

    try:
        # Query knowledge graph
        conjunctions = kg.get_conjunctions_by_satellite(
            satellite_id=satellite,
            min_probability=threshold,
            time_window_days=days
        )

        click.echo(f"Found {len(conjunctions)} potential conjunctions")

        if output:
            with open(output, 'w') as f:
                json.dump(conjunctions, f, indent=2, default=str)
            click.echo(f"✓ Saved to {output}")
        else:
            for conj in conjunctions:
                click.echo(f"\n{conj['secondary_name']}:")
                click.echo(f"  TCA: {conj['tca']}")
                click.echo(f"  Miss distance: {conj['miss_distance']:.2f} km")
                click.echo(f"  Probability: {conj['probability']:.2e}")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.group()
def serve():
    """Server and service commands"""
    pass


@serve.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--workers', default=4, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.pass_context
def api(ctx, host, port, workers, reload):
    """Start the API server"""
    import uvicorn

    click.echo(f"Starting API server on {host}:{port}...")
    click.echo(f"Workers: {workers}")

    uvicorn.run(
        "space_debris_tracker.api.server:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level="info"
    )


@serve.command()
@click.option('--port', default=8501, help='Port to bind to')
@click.pass_context
def dashboard(ctx, port):
    """Start the Streamlit dashboard"""
    import subprocess

    click.echo(f"Starting dashboard on port {port}...")

    dashboard_path = Path(__file__).parent.parent / 'dashboard' / 'app.py'

    subprocess.run([
        'streamlit', 'run',
        str(dashboard_path),
        '--server.port', str(port),
        '--server.headless', 'true'
    ])


@cli.group()
def monitor():
    """Monitoring and agent commands"""
    pass


@monitor.command()
@click.option('--satellites', required=True, help='Comma-separated list of NORAD IDs')
@click.option('--update-rate', default=60, help='Update rate in seconds')
@click.pass_context
def start_agent(ctx, satellites, update_rate):
    """Start monitoring agent for satellites"""
    from space_debris_tracker.monitoring_agents import SpaceMonitoringAgent

    sat_ids = [int(s.strip()) for s in satellites.split(',')]

    click.echo(f"Starting monitoring agents for {len(sat_ids)} satellites...")

    async def run_agents():
        agents = []
        for sat_id in sat_ids:
            agent = SpaceMonitoringAgent(
                satellite_id=sat_id,
                operator_preferences=ctx.obj['config'].get('monitoring', {})
            )
            agents.append(agent)

        # Start all agents
        tasks = [agent.continuous_monitoring() for agent in agents]
        await asyncio.gather(*tasks)

    try:
        asyncio.run(run_agents())
    except KeyboardInterrupt:
        click.echo("\n✓ Agents stopped")


@cli.command()
def version():
    """Show version information"""
    from space_debris_tracker import __version__
    click.echo(f"Space Debris Tracker v{__version__}")


@cli.command()
def status():
    """Check system status"""
    import torch

    click.echo("Space Debris Tracking System Status")
    click.echo("=" * 50)

    # Check Python version
    click.echo(f"Python: {sys.version.split()[0]}")

    # Check PyTorch
    click.echo(f"PyTorch: {torch.__version__}")
    click.echo(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"CUDA Version: {torch.version.cuda}")
        click.echo(f"GPU: {torch.cuda.get_device_name(0)}")
        click.echo(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Check database connections
    click.echo("\nDatabase Connections:")
    try:
        from space_debris_tracker.knowledge_graph import SpaceKnowledgeGraph
        kg = SpaceKnowledgeGraph()
        click.echo("  Neo4j: ✓ Connected")
    except:
        click.echo("  Neo4j: ✗ Not connected")

    # Check Kafka
    try:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(bootstrap_servers=['localhost:9092'])
        click.echo("  Kafka: ✓ Connected")
        consumer.close()
    except:
        click.echo("  Kafka: ✗ Not connected")


def main():
    """Entry point for the CLI"""
    cli(obj={})


if __name__ == '__main__':
    main()
