import argparse
import os
import sys
import numpy as np
import torch
import random
import warnings
warnings.filterwarnings('ignore')

# Import the required functions from your utils
from utils import create_land_based_spatial_block_cv, spatial_cross_blocks_scan

def load_benchmark_data(results_path, coords_path):
    """
    Load the benchmark data files.

    Args:
        results_path: Path to all_results.npy file
        coords_path: Path to all_coords.npy file

    Returns:
        all_results: Species occurrence data
        all_coords: Coordinate data
    """
    print(f"Loading data files...")
    print(f"  Results: {results_path}")
    print(f"  Coordinates: {coords_path}")

    try:
        all_results = np.load(results_path)
        all_coords = np.load(coords_path)

        print(f"Data loaded successfully:")
        print(f"  Results shape: {all_results.shape}")
        print(f"  Coordinates shape: {all_coords.shape}")

        # Basic validation
        if len(all_results) != len(all_coords):
            raise ValueError("Number of samples in results and coordinates must match")

        # Ensure data is in proper format for BCE loss
        if all_results.max() > 1.0:
            print("  Converting to binary format (0/1) for BCE loss")
            all_results = (all_results > 0).astype(np.float32)

        return all_results, all_coords

    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def setup_spatial_cv(all_coords, n_folds=5, block_size=20, margin=1, seed=42):
    """
    Set up spatial block cross-validation.

    Args:
        all_coords: Coordinate data
        n_folds: Number of cross-validation folds
        block_size: Size of spatial blocks in degrees
        margin: Margin size in degrees for buffer between train/test
        seed: Random seed for reproducibility

    Returns:
        train_masks, eval_masks, blocks, land_grid_indices
    """
    print(f"\nSetting up spatial cross-validation:")
    print(f"  Folds: {n_folds}")
    print(f"  Block size: {block_size}°")
    print(f"  Margin: {margin}°")
    print(f"  Random seed: {seed}")

    train_masks, eval_masks, blocks, land_grid_indices = create_land_based_spatial_block_cv(
        all_coords,
        n_folds=n_folds,
        block_size=block_size,
        margin=margin,
        seed=seed
    )

    # Print fold statistics
    print(f"\nFold statistics:")
    for i in range(n_folds):
        n_train = np.sum(train_masks[i])
        n_eval = np.sum(eval_masks[i])
        print(f"  Fold {i+1}: {n_train:,} train, {n_eval:,} eval samples")

    return train_masks, eval_masks, blocks, land_grid_indices

def run_benchmark(all_results, all_coords, train_masks, eval_masks, params, save_dir, save_model=False):
    """
    Run the spatial cross-validation benchmark.

    Args:
        all_results: Species occurrence data
        all_coords: Coordinate data
        train_masks: Training masks for each fold
        eval_masks: Evaluation masks for each fold
        params: Dictionary of hyperparameters
        save_dir: Directory to save results
        save_model: Whether to save trained models

    Returns:
        DataFrame with results
    """
    print(f"\nStarting spatial cross-validation benchmark...")
    print(f"Save directory: {save_dir}")
    print(f"Save models: {save_model}")

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Print parameter configuration
    print(f"\nHyperparameter configuration:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Run the spatial cross-validation scan
    results_df = spatial_cross_blocks_scan(
        all_results,
        all_coords,
        train_masks,
        eval_masks,
        params,
        save_model=save_model,
        save_dir=save_dir
    )

    print(f"\nBenchmark complete!")
    print(f"Results saved to: {save_dir}")

    return results_df

def print_summary_statistics(results_df):
    """
    Print summary statistics from the benchmark results.

    Args:
        results_df: DataFrame with benchmark results
    """
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    # Overall metrics
    metrics_to_summarize = [
        'region_AUROC', 'region_AUPRC', 'region_MSE',
        'region_Species_Pearson', 'region_Species_Spearman',
        'region_Sample_Pearson', 'region_Sample_Spearman',
        'region_TSS'
    ]

    print("\nOverall Performance Metrics (mean ± std):")
    for metric in metrics_to_summarize:
        if metric in results_df.columns:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            print(f"  {metric.replace('region_', '')}: {mean_val:.4f} ± {std_val:.4f}")

    # Per-fold performance
    if 'fold' in results_df.columns:
        print("\nPer-Fold AUROC Performance:")
        for fold in sorted(results_df['fold'].unique()):
            fold_data = results_df[results_df['fold'] == fold]
            if 'region_AUROC' in fold_data.columns:
                mean_auroc = fold_data['region_AUROC'].mean()
                print(f"  Fold {fold+1}: {mean_auroc:.4f}")

    # Training statistics
    print("\nTraining Statistics:")
    if 'training_time' in results_df.columns:
        total_time = results_df['training_time'].sum()
        avg_time = results_df['training_time'].mean()
        print(f"  Total training time: {total_time:.2f} seconds")
        print(f"  Average time per model: {avg_time:.2f} seconds")

    if 'num_params' in results_df.columns:
        num_params = results_df['num_params'].iloc[0]
        print(f"  Model parameters: {num_params:,}")

def main():
    parser = argparse.ArgumentParser(description="Spatial Cross-Validation Benchmark for EcoVAE")

    # Data arguments
    parser.add_argument("--results", type=str, default="./data/benchmarking/benchmark_input.npy",
                        help="Path to all_results.npy file")
    parser.add_argument("--coords", type=str, default="./data/benchmarking/benchmark_coords.npy",
                        help="Path to all_coords.npy file")

    # Cross-validation arguments
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--block_size", type=int, default=20,
                        help="Size of spatial blocks in degrees (default: 20)")
    parser.add_argument("--margin", type=float, default=1,
                        help="Margin size in degrees for buffer (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    # Model hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=2048,
                        help="Hidden dimension (default: 2048)")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Latent dimension (default: 32)")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Batch size (default: 2048)")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--masking_ratio", type=float, default=0.5,
                        help="Masking ratio (default: 0.5)")
    parser.add_argument("--lambda_weight", type=float, default=0.5,
                        help="Lambda weight for masked loss (default: 0.5)")
    parser.add_argument("--kl_weight", type=float, default=0.0,
                        help="KL divergence weight (default: 0.0)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--loss_function", type=str, default="bce_kl",
                        choices=["bce_kl", "mse_kl", "huber_kl"],
                        help="Loss function to use (default: bce_kl)")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "gelu", "elu"],
                        help="Activation function (default: relu)")

    # Output arguments
    parser.add_argument("--save_dir", type=str, default="./data/benchmarking/",
                        help="Directory to save results (default: ./data/benchmarking/)")
    parser.add_argument("--save_model", action="store_true",
                        help="Save trained models")
    parser.add_argument("--print_summary", action="store_true", default=True,
                        help="Print summary statistics after benchmark")

    args = parser.parse_args()

    # Load data
    all_results, all_coords = load_benchmark_data(args.results, args.coords)

    # Setup spatial cross-validation
    train_masks, eval_masks, blocks, land_grid_indices = setup_spatial_cv(
        all_coords,
        n_folds=args.n_folds,
        block_size=args.block_size,
        margin=args.margin,
        seed=args.seed
    )

    # Prepare hyperparameters
    scan_params = {
        "hidden_dims": args.hidden_dim,
        "latent_dims": args.latent_dim,
        "batch_sizes": args.batch_size,
        "learning_rates": args.learning_rate,
        "masking_ratios": args.masking_ratio,
        "lambda_weights": args.lambda_weight,
        "kl_weights": args.kl_weight,
        "epoch": args.epochs,
        "loss_functions": args.loss_function,
        "activations": args.activation
    }

    # Run benchmark
    results_df = run_benchmark(
        all_results,
        all_coords,
        train_masks,
        eval_masks,
        scan_params,
        args.save_dir,
        args.save_model
    )

    # Print summary if requested
    if args.print_summary and results_df is not None:
        print_summary_statistics(results_df)

    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    main()
