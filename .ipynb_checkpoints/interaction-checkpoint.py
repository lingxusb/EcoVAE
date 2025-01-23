import torch
import numpy as np
import argparse
from model import VAE
from tqdm import tqdm

# Function to apply the model in batches and return the reconstructed data
def apply_model(model, data, batch_size=2048, device='cuda'):
    model.eval()
    num_samples = data.size(0)
    recon_x_list = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = data[i:i + batch_size].to(device)
            batch_recon_x, _, _ = model(batch)
            recon_x_list.append(batch_recon_x.cpu())

    return torch.cat(recon_x_list, dim=0)

# Function to save results to a specified file
def save_results(results, file_path):
    np.save(file_path, results)

# Main function to load model, apply calculations, and save results
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = VAE(args.input_dim, args.hidden_dim, args.latent_dim, args.input_dim).to(device)
    state_dict = torch.load(f'./model/{args.model}', map_location=device)
    model.load_state_dict(state_dict)

    # Load data
    x_eval = torch.FloatTensor(np.load(f'./data/interpolation/{args.input}')).to(device)

    # Load global genus indices
    global_genus_idx = np.load("./data/interaction/genus_list.npy", allow_pickle=True).item()
    idxs = np.where(np.sum(x_eval.cpu().numpy(), axis=0) > 0)[0]
    print(f"number of genus to scan: {len(idxs)}")

    mutant_values = []
    mutant_background = []
    mutant_genus = []
    mutant_num = []

    num_samples = x_eval.size(0)

    # Iterate over the selected indices with a progress bar
    for k in tqdm(range(len(idxs)), desc="Processing genus", ncols=100):  # tqdm progress bar added here
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

    print("Results saved to ./data/interaction/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply VAE model to input data and perform mutant analysis")
    
    # Model and file-related arguments
    parser.add_argument("-m", "--model", type=str, default="model_all_regions.pth",
                        help="Name of the model file (default: model_all_regions.pth)")
    parser.add_argument("-i", "--input", type=str, default="interpolation_input.npy",
                        help="Name of the input file (default: interpolation_input.npy)")

    # Hyperparameter arguments
    parser.add_argument("-id", "--input_dim", type=int, default=13125, help="Input dimension of the model")
    parser.add_argument("-hd", "--hidden_dim", type=int, default=256, help="Hidden dimension of the model")
    parser.add_argument("-ld", "--latent_dim", type=int, default=32, help="Latent dimension of the model")
    
    # Batch size argument
    parser.add_argument("-b", "--batch_size", type=int, default=2048, help="Batch size for processing the input data")
    
    args = parser.parse_args()

    # Run main function
    main(args)
