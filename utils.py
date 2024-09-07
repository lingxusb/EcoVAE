import torch
import torch.nn as nn

def evaluate_model(model, x_eval, device):
    model.eval()
    with torch.no_grad():
        x_eval = x_eval.to(device)
        recon_x, _, _ = model(x_eval)
        recon_loss = nn.MSELoss(reduction='none')(recon_x, x_eval).sum(dim=-1).mean()
    return recon_loss.item()