import argparse
import sys
import os
import yaml

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.solver import ExperimentSolver, InferenceSolver, EvalSolver

def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}, using defaults.")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Multi-Task Solver for Video Agent")
    parser.add_argument("--task", type=str, required=True, choices=["experiment", "inference", "eval"], help="Task to run")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    
    # Optional arguments
    parser.add_argument("--video_path", type=str, default="data/raw/demo.mp4", help="Video path for inference")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for training")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples for evaluation")
    
    args = parser.parse_args()
    
    # Load base config
    config = load_config(args.config)
    
    # Update with command line args
    if hasattr(args, 'video_path'): config['video_path'] = args.video_path
    if hasattr(args, 'epochs'): config['epochs'] = args.epochs
    if hasattr(args, 'num_samples'): config['num_samples'] = args.num_samples
    
    print(f"Starting Task: {args.task}")

    if args.task == "experiment":
        solver = ExperimentSolver(config)
    elif args.task == "inference":
        solver = InferenceSolver(config)
    elif args.task == "eval":
        solver = EvalSolver(config)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    solver.run()

if __name__ == "__main__":
    main()
