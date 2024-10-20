import torch 
import numpy as np
import argparse
from model import VAE  

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
    # Save results to npy file
    np.save(file_path, results.numpy())

def main(model_name, input_file, output_file, input_dim, hidden_dim, latent_dim, batch_size):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model_path = f'./model/{model_name}'
    model = VAE(input_dim, hidden_dim, latent_dim, input_dim).to(device)

    # Load the state dict with appropriate map_location
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model = model.to(device)

    # Read input data
    data = np.load(f'./data/interpolation/{input_file}')
    input_data = torch.FloatTensor(data)

    # Apply the model
    results = apply_model(model, input_data, batch_size=batch_size, device=device)

    # Save results
    save_results(results, f'./data/interpolation/{output_file}')

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply VAE model to input data")
    
    # Arguments for model and input/output files
    parser.add_argument("-m","--model", type=str, default="model_all_regions.pth",
                        help="Name of the model file (default: model_all_regions.pth)")
    parser.add_argument("-i","--input", type=str, default="interpolation_input.npy",
                        help="Name of the input file (default: interpolation_input.npy)")
    parser.add_argument("-o","--output", type=str, default="interpolation_results.npy",
                        help="Name of the output file to save the results")
    
    # Arguments for hyperparameters
    parser.add_argument("-id","--input_dim", type=int, default=13125, help="Input dimension of the model")
    parser.add_argument("-hd","--hidden_dim", type=int, default=256, help="Hidden dimension of the model")
    parser.add_argument("-ld","--latent_dim", type=int, default=32, help="Latent dimension of the model")
    
    # Argument for batch size
    parser.add_argument("-b","--batch_size", type=int, default=2048, help="Batch size for processing the input data")
    
    args = parser.parse_args()

    # Call main function with the parsed arguments
    main(args.model, args.input, args.output, args.input_dim, args.hidden_dim, args.latent_dim, args.batch_size)
