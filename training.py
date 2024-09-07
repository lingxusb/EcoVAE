import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import VAE, loss_function
from data_loader import load_data
from utils import evaluate_model

def train(model, train_loader, x_eval, optimizer, device, lambda_weight, masking_ratio, num_epochs):
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
        PATH = './model/model.pth'
        torch.save(model.state_dict(), PATH)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    x_train, x_eval = load_data()
    
    # Create model
    input_dim = x_train.shape[1]
    model = VAE(input_dim, args.hidden_dim, args.latent_dim, input_dim).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create data loader
    train_dataset = TensorDataset(x_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Train the model
    train(model, train_loader, x_eval, optimizer, device, args.lambda_weight, args.masking_ratio, args.num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE Training")
    parser.add_argument("-e", "--num_epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--masking_ratio", type=float, default=0.5, help="Masking ratio")
    parser.add_argument("--lambda_weight", type=float, default=0.5, help="Lambda weight")
    
    args = parser.parse_args()
    main(args)