import torch
import torch.nn as nn
import numpy as np
import argparse
from model import VAE
from tqdm import tqdm

def apply_model(model, data, batch_size=2048, device='cuda'):
    """Apply the trained model to new data in batches."""
    model.eval()
    num_samples = data.size(0)
    recon_x_list = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = data[i:i + batch_size].to(device)
            batch_recon_x, _, _ = model(batch)
            recon_x_list.append(batch_recon_x.cpu())

    return torch.cat(recon_x_list, dim=0)

def save_results(results, file_path):
    """Save results to a specified file."""
    np.save(file_path, results)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set activation function
    if args.activation == 'gelu':
        activation_fn = nn.GELU()
    elif args.activation == 'relu':
        activation_fn = nn.ReLU()
    elif args.activation == 'elu':
        activation_fn = nn.ELU()
    else:
        activation_fn = nn.ReLU()  # Default

    # Load the model
    model = VAE(args.input_dim, args.hidden_dim, args.latent_dim, args.input_dim, activation_fn).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(f'./model/{args.model}', map_location=device)
    
    # Handle both old and new save formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model with saved hyperparameters")
            if 'hyperparameters' in checkpoint:
                print(f"Model hyperparameters: {checkpoint['hyperparameters']}")
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    # Load data
    x_eval_np = np.load(f'./data/interpolation/{args.input}')
    
    # Ensure data is in [0, 1] range for sigmoid output
    if x_eval_np.max() > 1.0:
        print("Normalizing input data to [0, 1] range")
        x_eval_np = (x_eval_np > 0).astype(np.float32)
    
    x_eval = torch.FloatTensor(x_eval_np).to(device)

    # Load global genus indices
    global_genus_idx = np.load("./data/interaction/genus_list.npy", allow_pickle=True).item()
    idxs = np.where(np.sum(x_eval.cpu().numpy(), axis=0) > 0)[0]
    print(f"Number of genus to scan: {len(idxs)}")

    mutant_values = []
    mutant_background = []
    mutant_genus = []
    mutant_num = []

    num_samples = x_eval.size(0)

    # Iterate over the selected indices with a progress bar
    for k in tqdm(range(len(idxs)), desc="Processing genus", ncols=100):
        idx = idxs[k]
        genus_name = [key for key, value in global_genus_idx.items() if value == idx]
        mutant_genus.append(genus_name)

        # Clone x_eval and set the index in the clone to 1
        x_eval_mutant = x_eval.clone().detach()
        x_eval_mutant[:, idx] = 1

        # Apply the model to both the original and mutated data
        recon_x_list_original = []
        recon_x_list_mutant = []

        with torch.no_grad():
            for i in range(0, num_samples, args.batch_size):
                batch = x_eval[i:i + args.batch_size].to(device)
                batch_recon_x, _, _ = model(batch)
                recon_x_list_original.append(batch_recon_x.cpu())

            for i in range(0, num_samples, args.batch_size):
                batch_mutant = x_eval_mutant[i:i + args.batch_size].to(device)
                batch_recon_x_mutant, _, _ = model(batch_mutant)
                recon_x_list_mutant.append(batch_recon_x_mutant.cpu())

        # Concatenate all batch outputs
        recon_x_original = torch.cat(recon_x_list_original, dim=0)
        recon_x_mutant = torch.cat(recon_x_list_mutant, dim=0)

        # Identify the positions where the original x_eval value is 0
        idx_tmp = np.where(x_eval.cpu().numpy()[:, idx] == 0)[0]

        # Convert reconstructed data to numpy and sum over selected positions
        recon_x_clone = recon_x_original.cpu().numpy()[:, idxs]
        recon_x_mutant_clone = recon_x_mutant.cpu().numpy()[:, idxs]
        mutant_values.append(np.sum(recon_x_mutant_clone[idx_tmp, :], axis=0))
        mutant_background.append(np.sum(recon_x_clone[idx_tmp, :], axis=0))
        mutant_num.append(len(idx_tmp))

    # Save the calculated results
    np.savetxt('./data/interaction/interaction_background.txt', np.array(mutant_background))
    np.savetxt('./data/interaction/interaction_addition.txt', np.array(mutant_values))
    np.savetxt('./data/interaction/interaction_genus.txt', np.array(mutant_genus), fmt="%s")
    np.savetxt('./data/interaction/interaction_num.txt', np.array(mutant_num))

    print("Results saved to ./data/interaction/")
    print(f"Processed {len(idxs)} genera")
    print(f"Average number of affected samples per genus: {np.mean(mutant_num):.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply VAE model to input data and perform mutant analysis")
    
    # Model and file-related arguments
    parser.add_argument("-m", "--model", type=str, default="model_all_regions.pth",
                        help="Name of the model file (default: model_all_regions.pth)")
    parser.add_argument("-i", "--input", type=str, default="interpolation_input.npy",
                        help="Name of the input file (default: interpolation_input.npy)")

    # Hyperparameter arguments - updated defaults
    parser.add_argument("-id", "--input_dim", type=int, default=11555, 
                        help="Input dimension of the model")
    parser.add_argument("-hd", "--hidden_dim", type=int, default=2048, 
                        help="Hidden dimension of the model (default: 2048)")
    parser.add_argument("-ld", "--latent_dim", type=int, default=32, 
                        help="Latent dimension of the model (default: 32)")
    parser.add_argument("-a", "--activation", type=str, default='relu',
                        choices=['relu', 'gelu', 'elu'],
                        help="Activation function (default: relu)")
    
    # Batch size argument
    parser.add_argument("-b", "--batch_size", type=int, default=2048, 
                        help="Batch size for processing the input data")
    
    args = parser.parse_args()

    # Run main function
    main(args)
