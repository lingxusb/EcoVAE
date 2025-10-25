import torch
import torch.nn as nn
import numpy as np
import argparse
from model import VAE

def apply_model(model, data, batch_size=2048, device='cuda'):
    """Apply the trained model to new data in batches."""
    model.eval()
    num_samples = data.size(0)
    recon_x_list = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = data[i:i+batch_size].to(device)
            batch_recon_x, _, _ = model(batch)
            recon_x_list.append(batch_recon_x.cpu())

    return torch.cat(recon_x_list, dim=0)

def save_results(results, file_path):
    """Save results to npy file."""
    np.save(file_path, results.numpy())

def main(model_name, input_file, output_file, input_dim, hidden_dim, latent_dim, 
         batch_size, activation='relu'):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model_path = f'./model/{model_name}'
    
    # Set activation function
    if activation == 'gelu':
        activation_fn = nn.GELU()
    elif activation == 'relu':
        activation_fn = nn.ReLU()
    elif activation == 'elu':
        activation_fn = nn.ELU()
    else:
        activation_fn = nn.ReLU()  # Default
    
    # Create model
    model = VAE(input_dim, hidden_dim, latent_dim, input_dim, activation_fn).to(device)
    
    # Load the saved checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both old and new save formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model with saved hyperparameters")
            if 'hyperparameters' in checkpoint:
                print(f"Model hyperparameters: {checkpoint['hyperparameters']}")
        else:
            # Old format - just state dict
            model.load_state_dict(checkpoint)
    else:
        # Very old format - direct state dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)

    # Read input data
    data = np.load(f'./data/interpolation/{input_file}')
    
    # Ensure data is in [0, 1] range for sigmoid output
    if data.max() > 1.0:
        print("Normalizing input data to [0, 1] range")
        data = (data > 0).astype(np.float32)
    
    input_data = torch.FloatTensor(data)

    # Apply the model
    print(f"Processing {len(input_data)} samples...")
    results = apply_model(model, input_data, batch_size=batch_size, device=device)

    # Save results
    save_results(results, f'./data/interpolation/{output_file}')
    print(f"Results saved to ./data/interpolation/{output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply VAE model to input data for interpolation")
    
    # Arguments for model and input/output files
    parser.add_argument("-m", "--model", type=str, default="model_all_regions.pth",
                        help="Name of the model file (default: model_all_regions.pth)")
    parser.add_argument("-i", "--input", type=str, default="interpolation_input.npy",
                        help="Name of the input file (default: interpolation_input.npy)")
    parser.add_argument("-o", "--output", type=str, default="interpolation_results.npy",
                        help="Name of the output file to save the results")
    
    # Arguments for hyperparameters - updated defaults
    parser.add_argument("-id", "--input_dim", type=int, default=11555, 
                        help="Input dimension of the model")
    parser.add_argument("-hd", "--hidden_dim", type=int, default=2048, 
                        help="Hidden dimension of the model (default: 2048)")
    parser.add_argument("-ld", "--latent_dim", type=int, default=32, 
                        help="Latent dimension of the model (default: 32)")
    parser.add_argument("-a", "--activation", type=str, default='relu',
                        choices=['relu', 'gelu', 'elu'],
                        help="Activation function (default: relu)")
    
    # Argument for batch size
    parser.add_argument("-b", "--batch_size", type=int, default=2048, 
                        help="Batch size for processing the input data")
    
    args = parser.parse_args()

    # Call main function with the parsed arguments
    main(args.model, args.input, args.output, args.input_dim, args.hidden_dim, 
         args.latent_dim, args.batch_size, args.activation)