from colorama import init, Fore, Style
from modules.config import DetectorConfigs
import matplotlib.pyplot as plt
import torch

cfg = DetectorConfigs()

init()

# Optimizer
def optimizer(model): 
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )

def lr_scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

def log_epoch(epoch, avg_loss):
    print(f"{Fore.BLUE}{Style.NORMAL}Epoch {Fore.GREEN}{Style.BRIGHT}{epoch+1}{Fore.WHITE}{Style.NORMAL}/{Fore.GREEN}{Style.BRIGHT}{cfg.num_epochs} "
          f"{Fore.BLUE}{Style.NORMAL}Average Loss {Fore.GREEN}{Style.BRIGHT}{avg_loss}{Fore.WHITE}{Style.NORMAL} ")

def save_plots(losses:list):
    plt.figure(figsize=(10, 5))
    plt.title("Model Loss During Training")
    plt.plot(losses, label="Model Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{cfg.training_output_dir}/loss_plot.png", bbox_inches='tight')
