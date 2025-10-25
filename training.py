import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import VAE, bce_kl_loss
from data_loader import load_data
from utils import evaluate_model
import os

def train(model, train_loader, x_eval, optimizer, device, lambda_weight, masking_ratio, 
          kl_weight, num_epochs, model_path, mode, climvar):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for x_batch, in train_loader:
            x_batch = x_batch.to(device)
            
            if mode == "presence":
                # Generate random mask for masking ratio of dimensions
                mask = torch.rand(x_batch.shape[0], x_batch.shape[1]) < masking_ratio
                mask = mask.float().to(device)

                # Forward pass with masked input
                recon_batch, mu, log_var = model(x_batch * (1 - mask))

                # Compute BCE + KL loss
                loss = bce_kl_loss(recon_batch, x_batch, mu, log_var, mask, lambda_weight, kl_weight)
            else:
                mask = torch.rand(x_batch.shape[0], x_batch.shape[1]) < masking_ratio
                mask = mask.float().to(device)
                mask[:, -climvar:] = 0

                # Forward pass with masked input
                recon_batch, mu, log_var = model(x_batch * (1 - mask))

                # Compute BCE + KL loss
                loss = bce_kl_loss(recon_batch, x_batch[:, :-climvar], mu, log_var, mask[:, :-climvar], lambda_weight, kl_weight)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")
        
        # Evaluate the model
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            eval_metrics = evaluate_model(model, x_eval, device, masking_ratio, lambda_weight, kl_weight, mode, climvar)
            print(f"Eval Metrics - MSE: {eval_metrics['MSE']:.4f}, AUROC: {eval_metrics['AUROC']:.4f}")
    
    # Save model with specified path or default path
    if model_path:
        save_path = model_path
    else:
        save_path = './model/model.pth'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save model state dict and hyperparameters
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': {
            'input_dim': model.encoder[0].in_features,
            'hidden_dim': model.encoder[0].out_features,
            'latent_dim': model.mu.out_features,
            'lambda_weight': lambda_weight,
            'kl_weight': kl_weight,
            'masking_ratio': masking_ratio
        }
    }, save_path)
    print(f"Model saved to {save_path}")

def load_dataset( mode, input_path=None):
    if input_path is None:
        print("Using default toy dataset...")
        return load_data()
    else:
        print(f"Loading data from {input_path}")
        try:
            # Load numpy data
            data = np.load(input_path)
            print(f"Loading data shape is {data.shape}")
            
            # Ensure data is in [0, 1] range for BCE loss
            if data.max() > 1.0:
                print("Normalizing data to [0, 1] range for BCE loss")
                data = (data > 0).astype(np.float32)
            
            # Get total number of samples
            total_samples = len(data)
            
            # Create random indices for train/test split
            np.random.seed(42)  # For reproducibility
            indices = np.random.permutation(total_samples)
            split_idx = int(total_samples * 0.9)  # 90% for training
            
            # Split the data
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            
            # Convert to torch tensors
            x_train = torch.tensor(data[train_indices], dtype=torch.float32)
            x_eval = torch.tensor(data[test_indices], dtype=torch.float32)
            
            print(f"Data split: {len(x_train)} training samples, {len(x_eval)} evaluation samples")
            return x_train, x_eval
            
        except Exception as e:
            print(f"Error loading data from {input_path}: {e}")
            print("Falling back to toy dataset...")
            return load_data()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data based on input argument
    x_train, x_eval = load_dataset(args.mode, args.input)
    
    # Set activation function
    if args.activation == 'gelu':
        activation_fn = nn.GELU()
    elif args.activation == 'relu':
        activation_fn = nn.ReLU()
    elif args.activation == 'elu':
        activation_fn = nn.ELU()
    else:
        activation_fn = nn.ReLU()  # Default to ReLU (best performing)
    
    # Create model
    input_dim = x_train.shape[1]
    if args.mode == 'climate':
        model = VAE(input_dim, args.hidden_dim, args.latent_dim, input_dim-args.climvar, activation_fn).to(device)
    else:
        model = VAE(input_dim, args.hidden_dim, args.latent_dim, input_dim, activation_fn).to(device)
    
    # Print model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in VAE: {num_params:,}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create data loader
    train_dataset = TensorDataset(x_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Train the model
    train(model, train_loader, x_eval, optimizer, device, args.lambda_weight, 
          args.masking_ratio, args.kl_weight, args.num_epochs, args.output,args.mode, args.climvar)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE Training with BCE Loss")
    parser.add_argument("-e", "--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("-hd", "--hidden_dim", type=int, default=2048, help="Hidden dimension (default: 2048)")
    parser.add_argument("-ld", "--latent_dim", type=int, default=32, help="Latent dimension (default: 32)")
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("-m", "--masking_ratio", type=float, default=0.5, help="Masking ratio")
    parser.add_argument("-lw", "--lambda_weight", type=float, default=0.5, help="Lambda weight for masked loss")
    parser.add_argument("-kl", "--kl_weight", type=float, default=0.0, help="KL divergence weight (default: 0)")
    parser.add_argument("-a", "--activation", type=str, default='relu', 
                        choices=['relu', 'gelu', 'elu'], help="Activation function (default: relu)")
    parser.add_argument("-i", "--input", type=str, default=None, 
                        help="Input npy file path containing training and evaluation data")
    parser.add_argument("-o", "--output", type=str, default=None, 
                        help="Output path for saving the trained model")
    parser.add_argument("--mode", type=str, default='presence',
                        choices=['presence', 'climate'],
                        help="Training mode: presence-only or climate-enhanced (default: presence)")
    parser.add_argument("--climvar", type=int, default=19, help="Number of climate variables")
    
    args = parser.parse_args()
    main(args)