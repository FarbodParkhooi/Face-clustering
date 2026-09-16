# train.py

from colorama import init, Fore, Style
from modules.dataset import FaceDataloader
from modules.config import DetectorConfigs
from models.detector import Detector
from modules import train_utils
import torch, os

def main():
    cfg = DetectorConfigs()
    all_losses = []

    # Creating needed directories
    os.makedirs(cfg.training_output_dir, exist_ok=True)
    os.makedirs(cfg.model_save_path, exist_ok=True)
    os.makedirs(cfg.sample_save_path, exist_ok=True)

    # Dataset and DataLoader
    train_dataloader = FaceDataloader()

    # Model
    model = Detector()
    
    if cfg.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    # Optimizer
    opt = train_utils.optimizer(model)
    # Learning rate scheduler 
    scheduler = train_utils.lr_scheduler(opt)

    # Training loop
    model.train()
    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        samples_seen = 0 
        for batch in train_dataloader:
            if len(batch) == 0:
                continue
            
            for image, boxes, labels in batch: 

                # Move to device
                image = image.unsqueeze(0).to(device)   # add batch dimension -> (1,C,H,W)
                boxes = boxes.to(device)                # (N,4)

                # Skip if no ground-truth boxes after augmentation
                if boxes.shape[0] == 0:
                    continue

                # Wrap boxes in a list (Detector expects list for batch)
                gt_boxes_list = [boxes]

                # Forward pass
                losses = model(image, gt_boxes_list)

                # Sum losses
                total_loss = (losses['rpn_cls'] + losses['rpn_reg'] +
                            losses['roi_cls'] + losses['roi_reg'])

                # Backward pass
                opt.zero_grad()
                total_loss.backward()
                opt.step()

                epoch_loss += total_loss.item()
                samples_seen += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, samples_seen)

        # Log after each epoch
        train_utils.log_epoch(epoch, avg_loss)

        # Updating losses list
        all_losses.append(avg_loss)

        # Save checkpoint
        if ((epoch+1) % cfg.save_checkpoint_every_epoch) == 0:
            torch.save(model.state_dict(), f"{cfg.model_save_path}/checkpoint_model_epoch_{epoch}.pth")
    
    # Save model
    torch.save(model.state_dict(), f"{cfg.model_save_path}/final_model.pth")
    print(f"Model saved to {cfg.model_save_path}")

    # Save plot
    train_utils.save_plots(all_losses)

    print(f"{Fore.GREEN}{Style.BRIGHT}\n\nTraining Completed Successfuly!\n\n{Fore.WHITE}{Style.NORMAL}")


if __name__ == "__main__":
    main()