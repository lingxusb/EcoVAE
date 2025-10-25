import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import pearsonr, spearmanr
from model import bce_kl_loss, VAE
from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon

import shapefile  # pyshp
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from itertools import product
import random
import gc

def _as_list(value):
    """Return value as list, so we can iterate uniformly."""
    if value is None:
        return []
    return value if isinstance(value, (list, tuple)) else [value]


def evaluate_model(model, x_eval, device, masking_ratio=0.5, lambda_weight=0.5, kl_weight=0.0, mode = "presence", climvar = 19):
    """
    Comprehensive evaluation of the VAE model with multiple metrics.

    Args:
        model: The VAE model to evaluate
        x_eval: Evaluation data
        device: Device to run evaluation on
        masking_ratio: Ratio of dimensions to mask
        lambda_weight: Weight for masked loss
        kl_weight: KL divergence weight

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()

    # Convert to tensor if needed
    if not isinstance(x_eval, torch.Tensor):
        x_eval = torch.tensor(x_eval, dtype=torch.float32)

    input_dim = x_eval.shape[1]

    with torch.no_grad():
        x_eval = x_eval.to(device)
        
        if mode == "presence":

            # Generate random mask for evaluation
            mask = torch.rand(x_eval.shape[0], input_dim) < masking_ratio
            mask = mask.float().to(device)

            # Apply mask to input
            masked_x = x_eval * (1 - mask)

            # Forward pass
            recon_x, mu, log_var = model(masked_x)

            # Calculate BCE + KL loss
            loss = bce_kl_loss(recon_x, x_eval, mu, log_var, mask, lambda_weight, kl_weight)

            # Calculate MSE for comparison
            mse_loss = nn.MSELoss(reduction='none')(recon_x, x_eval)
            mse_loss = (mse_loss * mask).sum() / mask.sum() if mask.sum() > 0 else mse_loss.mean()

            # Move to CPU for metric calculations
            x_eval_np = x_eval.cpu().numpy()
            recon_x_np = recon_x.cpu().numpy()
            mask_np = mask.cpu().numpy().astype(bool)
        else:
            # Generate random mask for evaluation
            mask = torch.rand(x_eval.shape[0], input_dim) < masking_ratio
            mask = mask.float().to(device)
            mask[:, -climvar:] = 0
            # Apply mask to input
            masked_x = x_eval * (1 - mask)

            # Forward pass
            recon_x, mu, log_var = model(masked_x)

            # Calculate BCE + KL loss
            loss = bce_kl_loss(recon_x, x_eval[:, :-climvar], mu, log_var, mask[:, :-climvar], lambda_weight, kl_weight)

            # Calculate MSE for comparison
            mse_loss = nn.MSELoss(reduction='none')(recon_x, x_eval[:, :-climvar])
            mse_loss = (mse_loss * mask[:, :-climvar]).sum() / mask[:, :-climvar].sum() if mask[:, :-climvar].sum() > 0 else mse_loss.mean()

            # Move to CPU for metric calculations
            x_eval_np = x_eval[:, :-climvar].cpu().numpy()
            recon_x_np = recon_x.cpu().numpy()
            mask_np = mask[:, :-climvar].cpu().numpy().astype(bool)

        # Calculate AUROC for masked values
        try:
            # Get masked values
            masked_true = x_eval_np[mask_np]
            masked_pred = recon_x_np[mask_np]

            # Convert to binary for AUROC calculation
            masked_true_binary = (masked_true > 0.5).astype(int)

            # Calculate AUROC
            if len(np.unique(masked_true_binary)) > 1:
                auroc = roc_auc_score(masked_true_binary, masked_pred)
            else:
                auroc = 0.5  # Default when only one class present
        except:
            auroc = 0.5

        # Calculate correlations (species-wise means)
        try:
            species_means_true = np.mean(x_eval_np, axis=0)
            species_means_pred = np.mean(recon_x_np, axis=0)

            species_pearson, _ = pearsonr(species_means_true, species_means_pred)
            species_spearman, _ = spearmanr(species_means_true, species_means_pred)
        except:
            species_pearson = 0.0
            species_spearman = 0.0

        # Calculate correlations (sample-wise means)
        try:
            sample_means_true = np.mean(x_eval_np, axis=1)
            sample_means_pred = np.mean(recon_x_np, axis=1)

            sample_pearson, _ = pearsonr(sample_means_true, sample_means_pred)
            sample_spearman, _ = spearmanr(sample_means_true, sample_means_pred)
        except:
            sample_pearson = 0.0
            sample_spearman = 0.0

        # Calculate F1 score for masked values
        try:
            threshold = 0.5  # Default threshold
            masked_pred_binary = (masked_pred > threshold).astype(int)
            f1 = f1_score(masked_true_binary, masked_pred_binary, average='macro', zero_division=0)
        except:
            f1 = 0.0

    return {
        'Loss': loss.item(),
        'MSE': mse_loss.item(),
        'AUROC': auroc,
        'Species_Pearson': species_pearson,
        'Species_Spearman': species_spearman,
        'Sample_Pearson': sample_pearson,
        'Sample_Spearman': sample_spearman,
        'F1_Score': f1
    }

def find_optimal_threshold(y_true, y_pred):
    """
    Find threshold that gives the same prevalence of 1's as in the original data.

    Args:
        y_true: Ground truth binary data
        y_pred: Predicted probabilities

    Returns:
        Optimal threshold value
    """
    original_prevalence = np.mean(y_true)

    # Try different thresholds
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0.5
    min_diff = float('inf')

    for t in thresholds:
        pred_binary = (y_pred > t).astype(int)
        pred_prevalence = np.mean(pred_binary)
        diff = abs(original_prevalence - pred_prevalence)

        if diff < min_diff:
            min_diff = diff
            best_threshold = t

    return best_threshold

def create_land_based_spatial_block_cv(all_coords, n_folds=5, block_size=20, margin=0, seed=42):
    """
    Create spatial block cross-validation masks focusing only on grids that overlap with land,
    with an option to add a margin around test blocks.

    Parameters:
    -----------
    all_coords : numpy array
        Array of shape (n_samples, 2) with longitude and latitude coordinates
    n_folds : int
        Number of cross-validation folds
    block_size : int
        Size of the spatial blocks in degrees
    margin : float
        Margin size in degrees to remove from the edges of test blocks (creates buffer between train/test)
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    train_masks : list of numpy arrays
        List of boolean masks for training sets
    eval_masks : list of numpy arrays
        List of boolean masks for evaluation sets (with margins applied)
    blocks : dict
        Dictionary with grid information
    land_grid_indices : list
        List of grid indices that overlap with land
    """
    np.random.seed(seed)
    random.seed(seed)

    # Define grid boundaries
    lon_min, lon_max = -180, 180
    lat_min, lat_max = -60, 80

    # Calculate number of blocks in each dimension
    n_lon_blocks = int((lon_max - lon_min) / block_size)
    n_lat_blocks = int((lat_max - lat_min) / block_size)

    # Load world map shapefile to identify land areas
    shapefile_path = "./data/tutorial/ne_110m_land.shp"
    sf = shapefile.Reader(shapefile_path)

    # Create land geometry
    land_shapes = sf.shapes()
    land_polygons = []
    for shape in land_shapes:
        points = shape.points
        if len(points) > 2:  # Valid polygon needs at least 3 points
            land_polygons.append(ShapelyPolygon(points))

    # Combine all land polygons
    land_multipolygon = MultiPolygon(land_polygons)

    # Check which grid cells overlap with land
    land_grid_indices = []
    print("Identifying grid cells that overlap with land...")
    for i in range(n_lon_blocks):
        for j in range(n_lat_blocks):
            # Define block boundaries
            lon_start = lon_min + i * block_size
            lon_end = lon_start + block_size
            lat_start = lat_min + j * block_size
            lat_end = lat_start + block_size

            # Create shapely polygon for the grid cell
            block_polygon = ShapelyPolygon([
                (lon_start, lat_start),
                (lon_end, lat_start),
                (lon_end, lat_end),
                (lon_start, lat_end)
            ])

            # Check if the grid cell intersects with land
            if block_polygon.intersects(land_multipolygon):
                land_grid_indices.append((i, j))

    print(f"Found {len(land_grid_indices)} grid cells that overlap with land.")

    # Randomly assign land blocks to folds
    random.shuffle(land_grid_indices)
    fold_assignments = {}

    for idx, block_idx in enumerate(land_grid_indices):
        fold_assignments[block_idx] = idx % n_folds

    # Store block boundaries for each point
    point_block_info = []
    for coord in all_coords:
        lon, lat = coord

        # Handle edge cases and normalize coordinates
        if lon < lon_min:
            lon += 360
        if lon >= lon_max:
            lon -= 360

        # Calculate block indices
        block_i = int((lon - lon_min) / block_size)
        block_j = int((lat - lat_min) / block_size)

        # Handle edge cases
        if block_i >= n_lon_blocks:
            block_i = n_lon_blocks - 1
        if block_j >= n_lat_blocks:
            block_j = n_lat_blocks - 1
        if block_i < 0:
            block_i = 0
        if block_j < 0:
            block_j = 0

        # Calculate position within block (as ratio 0-1)
        rel_lon = (lon - (lon_min + block_i * block_size)) / block_size
        rel_lat = (lat - (lat_min + block_j * block_size)) / block_size

        # Store block indices and relative position
        point_block_info.append({
            'block_idx': (block_i, block_j),
            'rel_lon': rel_lon,
            'rel_lat': rel_lat
        })

    # Create masks for each fold
    train_masks = []
    eval_masks = []

    # Convert margin to relative size within block (0-1 scale)
    rel_margin = margin / block_size

    if margin > 0:
        print(f"Applying margin of {margin}° ({rel_margin:.3f} of block size) around evaluation blocks")

    eval_block_counts = []  # To track how many points in each eval block

    for fold in range(n_folds):
        eval_mask = np.zeros(len(all_coords), dtype=bool)
        eval_with_margin = np.zeros(len(all_coords), dtype=bool)
        eval_block_count = 0

        for i, point_info in enumerate(point_block_info):
            block_idx = point_info['block_idx']

            # Only consider blocks that are on land and have been assigned to a fold
            if block_idx in fold_assignments and fold_assignments[block_idx] == fold:
                eval_mask[i] = True
                eval_block_count += 1

                # Apply margin - include only points that are not within the margin from any edge
                rel_lon = point_info['rel_lon']
                rel_lat = point_info['rel_lat']

                if (rel_margin <= rel_lon <= (1 - rel_margin) and
                    rel_margin <= rel_lat <= (1 - rel_margin)):
                    eval_with_margin[i] = True

        # If margin is zero, use the original mask
        final_eval_mask = eval_with_margin if margin > 0 else eval_mask

        # Count how many points we have with and without margin
        if margin > 0:
            points_with_margin = np.sum(eval_with_margin)
            points_without_margin = np.sum(eval_mask)
            margin_reduction = 1 - (points_with_margin / points_without_margin) if points_without_margin > 0 else 0
            print(f"Fold {fold+1}: {points_with_margin} evaluation points after applying margin "
                  f"({points_without_margin} before, {margin_reduction:.1%} reduction)")

        train_mask = ~eval_mask
        train_masks.append(train_mask)
        eval_masks.append(final_eval_mask)
        eval_block_counts.append(eval_block_count)

    # Create block information for visualization
    blocks = {
        'lon_min': lon_min,
        'lon_max': lon_max,
        'lat_min': lat_min,
        'lat_max': lat_max,
        'block_size': block_size,
        'margin': margin,
        'n_lon_blocks': n_lon_blocks,
        'n_lat_blocks': n_lat_blocks,
        'assignments': fold_assignments,
        'n_folds': n_folds,
        'land_grid_indices': land_grid_indices,
        'eval_block_counts': eval_block_counts
    }

    return train_masks, eval_masks, blocks, land_grid_indices

def spatial_cross_blocks_scan(all_results, all_coords, train_masks, eval_masks, params, save_model=True, save_dir='model_results/'):
    # Create directory to save results if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Ensure save_dir ends with a slash
    if not save_dir.endswith('/'):
        save_dir += '/'

    input_dim = all_results.shape[1]

    # Extract and process parameters
    hidden_dims     = _as_list(params.get("hidden_dims",     256))
    latent_dims     = _as_list(params.get("latent_dims",     32))
    batch_sizes     = _as_list(params.get("batch_sizes",     512))
    learning_rates  = _as_list(params.get("learning_rates",  1e-3))
    masking_ratios  = _as_list(params.get("masking_ratios",  0.5))
    lambda_weights  = _as_list(params.get("lambda_weights",  0.5))
    kl_weights      = _as_list(params.get("kl_weights",      0.0))
    loss_functions  = _as_list(params.get("loss_functions",  "mse_kl"))
    activations     = _as_list(params.get("activations",     "relu"))
    num_epochs = params.get("epoch",     10)

    # Generate all combinations
    all_combinations = list(product(
        hidden_dims, latent_dims, batch_sizes, learning_rates,
        masking_ratios, lambda_weights, kl_weights, loss_functions, activations
    ))

    # If we have more than 100 combinations, randomly select 100
    if len(all_combinations) > 100:
        random.seed(42)  # For reproducibility
        param_combinations = random.sample(all_combinations, 100)
    else:
        param_combinations = all_combinations

    print(f"Testing {len(param_combinations)} hyperparameter combinations")

    # Prepare results dataframe
    results = []

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine training mode based on masks structure
    is_cross_validation = len(train_masks) == len(eval_masks)

    # Process each parameter combination
    for i, params in enumerate(param_combinations):
        hidden_dim, latent_dim, batch_size, lr, masking_ratio, lambda_weight, kl_weight, loss_fn_name, activation_name = params

        # Set activation function
        if activation_name == 'gelu':
            activation_fn = nn.GELU()
        elif activation_name == 'relu':
            activation_fn = nn.ReLU()
        elif activation_name == 'elu':
            activation_fn = nn.ELU()
        else:
            activation_fn = nn.GELU()  # Default

        # Select loss function
        if loss_fn_name == 'mse_kl':
            loss_function = mse_kl_loss
        elif loss_fn_name == 'bce_kl':
            loss_function = bce_kl_loss
        elif loss_fn_name == 'huber_kl':
            loss_function = huber_kl_loss
        else:
            loss_function = mse_kl_loss  # Default

        # Define evaluation structure based on the mode
        if is_cross_validation:
            # Cross-validation mode (multiple folds)
            fold_range = range(len(train_masks))
        else:
            # Single training set, multiple evaluation sets
            fold_range = [0]  # Just one "fold" with different evaluation regions

        # Process each fold
        for fold in fold_range:
            # Create training and evaluation data based on mode
            if is_cross_validation:
                x_train = all_results[train_masks[fold]]
                x_eval_sets = {"fold": all_results[eval_masks[fold]]}
                model_id = f"vae_h{hidden_dim}_l{latent_dim}_{loss_fn_name}_kl{kl_weight}_{activation_name}_lr{lr}_lw{lambda_weight}_fold{fold}"
            else:
                x_train = all_results[train_masks[0]]
                x_eval_sets = {str(q): all_results[eval_masks[q]] for q in range(len(eval_masks))}
                model_id = f"vae_h{hidden_dim}_l{latent_dim}_{loss_fn_name}_kl{kl_weight}_{activation_name}_lr{lr}_lw{lambda_weight}"

            print(f"\nTraining model {i+1}/{len(param_combinations)}: {model_id}")

            # Convert input data to PyTorch tensor if not already
            if not isinstance(x_train, torch.Tensor):
                x_train = torch.tensor(x_train, dtype=torch.float32)

            # Create VAE model
            model = VAE(input_dim, hidden_dim, latent_dim, input_dim, activation_fn).to(device)

            # Print the number of parameters in the model
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of parameters in the VAE model: {num_params}")

            # Create optimizer
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Create DataLoader
            train_dataset = TensorDataset(x_train)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

            # Train the VAE
            start_time = time.time()
            train_losses = []
            eval_losses = []

            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0
                num_batches = 0

                for x_batch, in train_loader:
                    x_batch = x_batch.float().to(device)

                    # Generate random mask for k dimensions
                    mask = torch.rand(x_batch.shape[0], input_dim) < masking_ratio
                    mask = mask.float().to(device)

                    # Forward pass
                    recon_batch, mu, log_var = model(x_batch * (1 - mask))

                    # Compute VAE loss
                    loss = loss_function(recon_batch, x_batch, mu, log_var, mask, lambda_weight, kl_weight)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                # Calculate average loss
                avg_epoch_loss = epoch_loss / num_batches
                train_losses.append(avg_epoch_loss)

                # Evaluate every few epochs
                if epoch % 5 == 0 or epoch == num_epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        # Get first evaluation set for validation during training
                        x_eval_first = list(x_eval_sets.values())[0]

                        # Sample a batch from evaluation set
                        sample_idx = np.random.choice(len(x_eval_first), size=min(1000, len(x_eval_first)), replace=False)
                        x_eval_sample = torch.tensor(x_eval_first[sample_idx], dtype=torch.float32).to(device)

                        # Random mask for evaluation
                        eval_mask = torch.rand(x_eval_sample.shape[0], input_dim) < masking_ratio
                        eval_mask = eval_mask.float().to(device)

                        # Forward pass
                        recon_x, mu, log_var = model(x_eval_sample * (1 - eval_mask))

                        # Compute loss
                        eval_loss = loss_function(recon_x, x_eval_sample, mu, log_var, eval_mask, lambda_weight, kl_weight).item()
                        eval_losses.append(eval_loss)

                    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_loss:.4f}, Eval Loss: {eval_loss:.4f}")

            training_time = time.time() - start_time
            print(f"Model training time: {training_time:.2f} seconds")

            if save_model:
                # Save the model
                model_path = os.path.join(save_dir, f"{model_id}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'eval_losses': eval_losses,
                    'hyperparameters': {
                        'hidden_dim': hidden_dim,
                        'latent_dim': latent_dim,
                        'batch_size': batch_size,
                        'learning_rate': lr,
                        'masking_ratio': masking_ratio,
                        'lambda_weight': lambda_weight,
                        'kl_weight': kl_weight,
                        'loss_function': loss_fn_name,
                        'activation': activation_name,
                        'num_epochs': num_epochs
                    }
                }, model_path)

            # Clean up tensors to free memory
            try:
                del recon_batch, mu, log_var, loss, x_batch, mask, x_eval_sample, eval_mask
            except:
                pass

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Evaluate model on each evaluation set
            print("Evaluating on test regions...")

            # Process each evaluation set
            for eval_id, (eval_key, eval_data) in enumerate(x_eval_sets.items()):
                # For single-fold mode, add region identifier to model_id
                eval_model_id = model_id if is_cross_validation else f"{model_id}_{eval_key}"

                # Evaluate model on this evaluation set
                region_metrics = evaluate_model_complete(model, eval_data, masking_ratio, eval_model_id, batch_size, save_dir, device)

                # Create result row for this evaluation
                result = {
                    'model_id': model_id,
                    'fold': fold,
                    'eval_region': eval_key,
                    'hidden_dim': hidden_dim,
                    'latent_dim': latent_dim,
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'masking_ratio': masking_ratio,
                    'lambda_weight': lambda_weight,
                    'kl_weight': kl_weight,
                    'loss_function': loss_fn_name,
                    'activation': activation_name,
                    'num_params': num_params,
                    'training_time': training_time,
                    'final_train_loss': train_losses[-1],
                    'final_eval_loss': eval_losses[-1]
                }

                # Add region-specific metrics
                for metric_name, value in region_metrics.items():
                    result[f'region_{metric_name}'] = value

                results.append(result)

                # Save results after each evaluation to avoid data loss
                pd.DataFrame(results).to_csv(save_dir + 'spatial_cv_results_bce.csv', index=False)

            print(f"Model {model_id} evaluation complete")

            # Plot training and evaluation losses
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Model {model_id} - Training and Evaluation Loss')
            plt.legend()
            plt.savefig(save_dir + f"{model_id}_loss.png")
            plt.close()

        # Clean up model and data after processing all folds for this parameter combination
        del model, optimizer, x_train, x_eval_sets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # # Create summary statistics
    final_df = pd.DataFrame(results)
    print("\nHyperparameter scanning complete.")
    #
    # # Save final results
    # final_df.to_csv(save_dir + 'spatial_cv_results_bce_final.csv', index=False)

    return final_df

def evaluate_model_complete(model, x_data, masking_ratio, model_id, batch_size, save_dir='model_results', device='cuda'):

    subsample = 2000
    input_dim = x_data.shape[1]
    np.random.seed(42)
    random.seed(42)
    # constraint the x size
    if input_dim > subsample:
        selected_indices_col = np.random.choice(input_dim, size=subsample, replace=False)

    # Convert input data to PyTorch tensor if not already
    if not isinstance(x_data, torch.Tensor):
        x_data = torch.tensor(x_data, dtype=torch.float32)
    # Generate random mask for evaluation, targeting columns
    random_columns = torch.rand(input_dim) < masking_ratio  # Random draw for columns
    x_data = x_data.cpu()

    # Process in batches
    num_samples = x_data.shape[0]
    test_MSE = []

    # Ensure the model is in evaluation mode
    model.eval()

    if input_dim > subsample:
        masked_original_binary = np.zeros([num_samples, subsample], dtype = np.float32)
        masked_recon_data = np.zeros([num_samples, subsample], dtype = np.float32)
    else:
        masked_original_binary = np.zeros(x_data.shape, dtype = np.float32)
        masked_recon_data = np.zeros(x_data.shape, dtype = np.float32)
    # Add error handling for NaN outputs from model
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            try:
                batch = x_data[i:i+batch_size].float().to(device)
                eval_mask = torch.tile(random_columns, (batch.shape[0], 1)).float().to(device)
                # Process the batch through the model
                batch_recon_x, _, _ = model(batch*(1-eval_mask))

                if input_dim > subsample:
                    masked_recon_data[i:i+batch_size] = (batch_recon_x*eval_mask).cpu().numpy()[:, selected_indices_col]
                    masked_original_binary[i:i+batch_size] = (batch*eval_mask).cpu().numpy()[:, selected_indices_col]
                else:
                    masked_recon_data[i:i+batch_size] = (batch_recon_x*eval_mask).cpu().numpy()
                    masked_original_binary[i:i+batch_size] = (batch*eval_mask).cpu().numpy()
                # Check for NaN values and skip this batch if found
                if torch.isnan(batch_recon_x).any():
                    print(f"Warning: NaN values found in model output for batch {i//batch_size}. Skipping batch.")
                    continue

                loss = nn.MSELoss(reduction='none')(batch_recon_x, batch)
                test_MSE.append(loss.mean(axis=1).cpu().numpy())
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                continue
    del batch_recon_x, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    test_MSE = np.hstack(test_MSE)

    # Handle NaN values in the data
    # Replace NaNs with zeros for binary conversion
    masked_original_binary = np.nan_to_num(masked_original_binary, nan=0.0)
    masked_recon_data = np.nan_to_num(masked_recon_data, nan=0.0)

    # Convert real count numbers to binary labels
    masked_original_binary = (masked_original_binary > 0).astype(int)

    # Create arrays to store metrics for each species
    if input_dim > subsample:
        random_columns = random_columns[selected_indices_col]
    n_features = int(sum(random_columns))
    species_indices = np.where(random_columns==1)[0]
    auroc = np.zeros(n_features)
    auprc = np.zeros(n_features)
    tss_values = np.zeros(n_features)

    # Calculate mean values for correlation metrics
    species_means_true = np.nanmean(masked_original_binary, axis=0)
    species_means_pred = np.nanmean(masked_recon_data, axis=0)
    sample_means_true = np.nanmean(masked_original_binary, axis=1)
    sample_means_pred = np.nanmean(masked_recon_data, axis=1)

    # Calculate AUROC for each dimension
    for i, index in enumerate(species_indices):
        try:
            auroc[i] = roc_auc_score(masked_original_binary[:, index],
                                    masked_recon_data[:, index])
        except ValueError:
            auroc[i] = np.nan

    # Calculate mean AUROC
    mean_auroc = np.nanmean(auroc)

    # Calculate AUPRC for each dimension
    for i, idx in enumerate(species_indices):
        y_true = masked_original_binary[:, idx]
        y_score = masked_recon_data[:, idx]  # predicted probabilities

        # If a column is all-0 or all-1, PR AUC is undefined → NaN
        if y_true.min() == y_true.max():
            auprc[i] = np.nan
            continue

        # Use average_precision_score
        try:
            auprc[i] = average_precision_score(y_true, y_score)
        except ValueError:
            auprc[i] = np.nan

    # Mean AUCPRC across all selected dimensions
    mean_auprc = np.nanmean(auprc)

    # Calculate mean squared error (MSE) for the masked values
    # Use our own implementation to avoid sklearn's strict checking
    mse = 0

    # Calculate Pearson and Spearman correlations (species-wise)
    try:
        species_pearson, species_pearson_p = pearsonr(
            species_means_true,
            species_means_pred
        )
    except Exception as e:
        print(f"Error calculating species_pearson: {e}")
        species_pearson = np.nan
        species_pearson_p = np.nan

    try:
        species_spearman, species_spearman_p = spearmanr(
            species_means_true,
            species_means_pred
        )
    except Exception as e:
        print(f"Error calculating species_spearman: {e}")
        species_spearman = np.nan
        species_spearman_p = np.nan

    # Calculate Pearson and Spearman correlations (sample-wise)
    try:
        sample_pearson, sample_pearson_p = pearsonr(
            sample_means_true,
            sample_means_pred
        )
    except Exception as e:
        print(f"Error calculating sample_pearson: {e}")
        sample_pearson = np.nan
        sample_pearson_p = np.nan

    try:
        sample_spearman, sample_spearman_p = spearmanr(
            sample_means_true,
            sample_means_pred
        )
    except Exception as e:
        print(f"Error calculating sample_spearman: {e}")
        sample_spearman = np.nan
        sample_spearman_p = np.nan

    f1 = 0
    species_presence_ratio_error = 0

    # Calculate TSS (True Skill Statistic)
    from sklearn.metrics import confusion_matrix

    for i, index in enumerate(species_indices):
        try:
            # Get the original binary values and prediction scores for this dimension
            original = masked_original_binary[:, index]
            predictions = masked_recon_data[:, index]

            # Skip if all values are the same (can't calculate meaningful TSS)
            if len(np.unique(original)) < 2:
                # print(f"Skipping TSS calculation for dimension {index}: All ground truth values are identical")
                tss_values[i] = np.nan
                continue

            # Calculate TSS across different thresholds
            thresholds = np.unique(np.quantile(predictions, np.linspace(0, 1, 101)))  # 101 thresholds from 0 to 1
            tss_at_thresholds = []

            for threshold in thresholds:
                # Apply the threshold to get binary predictions
                pred_binary = (predictions > threshold).astype(int)

                # Get confusion matrix
                try:
                    cm = confusion_matrix(original, pred_binary, labels=[0, 1])

                    # Only proceed if we have a proper 2x2 confusion matrix
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()

                        # Calculate sensitivity (TPR) and specificity (TNR)
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                        # Calculate TSS at this threshold
                        tss = sensitivity + specificity - 1
                        tss_at_thresholds.append(tss)
                    else:
                        tss_at_thresholds.append(-1)  # Invalid result
                except Exception as e:
                    print(f"Error in TSS calculation at threshold {threshold} for dimension {index}: {e}")
                    tss_at_thresholds.append(-1)  # Error result

            # If we have valid TSS values, find the maximum
            if tss_at_thresholds and max(tss_at_thresholds) > -1:
                max_tss = max(tss_at_thresholds)
                best_threshold = thresholds[np.argmax(tss_at_thresholds)]
                # print(f"Dimension {index}: Max TSS = {max_tss:.4f} at threshold = {best_threshold:.2f}")
                tss_values[i] = max_tss
            else:
                # print(f"No valid TSS values for dimension {index}")
                tss_values[i] = np.nan

        except Exception as e:
            print(f"Error calculating TSS for dimension {index}: {e}")
            tss_values[i] = np.nan

    # Calculate mean TSS across all dimensions
    mean_tss = np.nanmean(tss_values)

    if input_dim > subsample:
        species_idx = selected_indices_col[species_indices]
    else:
        species_idx = np.arange(len(species_indices))
    # Save per-species metrics
    species_metrics_df = pd.DataFrame({
        'species_idx': species_idx,
        'auroc': auroc,
        'auprc': auprc,
        'max_tss': tss_values
    })
    species_metrics_df.to_csv(f"{save_dir}{model_id}_species_metrics.csv", index=False)

    # Save species correlation data
    species_correlation_df = pd.DataFrame({
        'species_idx': np.arange(len(species_means_true)),
        'species_mean_true': species_means_true,
        'species_mean_pred': species_means_pred
    })
    species_correlation_df.to_csv(f"{save_dir}{model_id}_species_correlation_data.csv", index=False)

    # Save sample correlation data
    sample_correlation_df = pd.DataFrame({
        'sample_idx': np.arange(len(sample_means_true)),
        'sample_mean_true': sample_means_true,
        'sample_mean_pred': sample_means_pred
    })
    sample_correlation_df.to_csv(f"{save_dir}{model_id}_sample_correlation_data.csv", index=False)

    # Save summary metrics
    summary_metrics = {
        'AUROC': mean_auroc,
        'AUPRC': mean_auprc,
        'MSE': mse,
        'Species_Pearson': species_pearson,
        'Species_Spearman': species_spearman,
        'Sample_Pearson': sample_pearson,
        'Sample_Spearman': sample_spearman,
        'F1_Score': f1,
        'Presence_Ratio_Error': species_presence_ratio_error,
        'TSS': mean_tss
    }

    pd.DataFrame([summary_metrics]).to_csv(f"{save_dir}{model_id}_summary_metrics.csv", index=False)

    print(f"Metrics saved in {save_dir}")


    # Return computed metrics
    metrics = {
        'AUROC': mean_auroc,
        'AUPRC': mean_auprc,
        'MSE': mse,
        'Species_Pearson': species_pearson,
        'Species_Spearman': species_spearman,
        'Sample_Pearson': sample_pearson,
        'Sample_Spearman': sample_spearman,
        'F1_Score': f1,
        'Presence_Ratio_Error': species_presence_ratio_error,
        'TSS': mean_tss  # Add TSS to metrics
    }
    del masked_original_binary, masked_recon_data
    gc.collect()

    return metrics
