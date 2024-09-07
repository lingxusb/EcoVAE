import torch
import numpy as np
import argparse
from model import VAE  

def load_data(file_path):
    # Load data from txt file
    # Adjust this function based on your specific data format
    data = np.load(f'./data/{file_path}')
    return torch.FloatTensor(data)

def apply_model(model, data, batch_size=2048, device='cuda'):
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
    # Save results to txt file
    np.save(file_path, results.numpy())

def main(model_name, input_file):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set hyperparameters
    input_dim = 13125
    hidden_dim = 256
    latent_dim = 32
    output_dim = 13125

    # Load the model
    model_path = f'./model/{model_name}'
    model = VAE(input_dim, hidden_dim, latent_dim, output_dim).to(device)
        # Load the state dict with appropriate map_location
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model = model.to(device)

    # Read input data
    input_data = load_data(input_file)

    # Apply the model
    results = apply_model(model, input_data, device=device)

    # Save results
    output_file = f'./data/interpolation_results'
    save_results(results, output_file)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply VAE model to input data")
    parser.add_argument("--model", type=str, default="model_all_regions.pth",
                        help="Name of the model file (default: model_all_regions.pth)")
    parser.add_argument("--input", type=str, default="interpolation_input.npy",
                        help="Name of the input file (default: interpolation_input.npy)")
    
    args = parser.parse_args()

    main(args.model, args.input)