import argparse
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import VAE, loss_function
from data_loader import load_data
from utils import evaluate_model
import os
import pickle

def train(model, train_loader, x_eval, optimizer, device, lambda_weight, masking_ratio, num_epochs, model_path):
    for epoch in range(num_epochs):
        model.train()
        for x_batch, in train_loader:
            x_batch = x_batch.to(device)
            mask = torch.rand(x_batch.shape[0], x_batch.shape[1]) < masking_ratio
            mask = mask.float().to(device)
            
            recon_batch, mu, log_var = model(x_batch * (1 - mask))
            loss = loss_function(recon_batch, x_batch, mu, log_var, mask, lambda_weight)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
        # Evaluate the model
        eval_loss = evaluate_model(model, x_eval, device)
        print(f"Eval set MSE loss: {eval_loss:.4f}")
    
    # Save model with specified path or default path
    if model_path:
        save_path = model_path
    else:
        save_path = './model/model.pth'
    
    # Create directory if it doesn't exist
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_dataset(input_path=None):
    if input_path is None:
        print("Using default toy dataset...")
        return load_data()
    else:
        print(f"Loading data from {input_path}")
        try:
            # Load numpy data
            data = np.load(input_path)
            print(f"loading data shape is {data.shape}")
            
            # Get total number of samples
            total_samples = len(data)
            
            # Create random indices for train/test split
            indices = np.random.permutation(total_samples)
            split_idx = int(total_samples * 0.9)  # 80% for training
            
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
    x_train, x_eval = load_dataset(args.input)
    
    # Create model
    input_dim = x_train.shape[1]
    model = VAE(input_dim, args.hidden_dim, args.latent_dim, input_dim).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create data loader
    train_dataset = TensorDataset(x_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Train the model
    train(model, train_loader, x_eval, optimizer, device, args.lambda_weight, 
          args.masking_ratio, args.num_epochs, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE Training")
    parser.add_argument("-e", "--num_epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("-hd","--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("-ld","--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("-l","--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-b","--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("-m","--masking_ratio", type=float, default=0.5, help="Masking ratio")
    parser.add_argument("-lw","--lambda_weight", type=float, default=0.5, help="Lambda weight")
    parser.add_argument("-i","--input", type=str, default=None, 
                       help="Input npy file path containing training and evaluation data")
    parser.add_argument("-o","--output", type=str, default=None, 
                       help="Output path for saving the trained model")
    
    args = parser.parse_args()
    main(args)