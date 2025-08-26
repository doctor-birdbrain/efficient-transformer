import os

import csv
import torch
from torch.utils.data import DataLoader
import wandb


def train_and_log(
    model,
    train_dataset,
    val_dataset=None,
    num_steps=100,
    val_interval=10,
    batch_size=32,
    lr=1e-3,
    device=None,
    project_name="training-curve-demo",
    local_log_path="local_logs",
    run_name="trial_run"
):
    """
    Train a PyTorch model for a fixed number of iterations and log the training loss to W&B.
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create data loader
    dataloader = DataLoader(train_dataset, batch_size=batch_size)
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size)


    # Simple optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # W&B init
    wandb.init(project=project_name, name=run_name, config={
        "batch_size": batch_size,
        "learning_rate": lr,
        "iterations": num_steps
    })

    # Training loop
    model.train()
    step = 0
    data_iter = iter(dataloader)
    # The file with the training log
    os.makedirs(f"{local_log_path}", exist_ok=True)
    # The file with the validation log
    os.makedirs(f"{local_log_path}", exist_ok=True)
    while step < num_steps:
        # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        inputs, targets = batch["image"], batch["label"]
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

         # Validation step
        if val_dataset and ((step + 1) % val_interval == 0 or step == num_steps - 1):
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs, val_targets = val_batch["image"], val_batch["label"]
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs)
                    v_loss = criterion(val_outputs, val_targets)
                    val_loss += v_loss.item() * val_inputs.size(0)

                    preds = torch.argmax(val_outputs, dim=1)
                    correct += (preds == val_targets).sum().item()
                    total += val_targets.size(0)

            avg_val_loss = val_loss / total
            val_acc = correct / total

            metrics = {
                "step": step,
                "val/loss": avg_val_loss,
                "val/accuracy": val_acc,
                "train/loss": loss.item()
            }
            wandb.log(metrics)
            append_dict_to_csv(f"{local_log_path}/{run_name}_val", metrics)

            model.train()
        else:
            metrics = {
              "step": step,
              "loss": loss.item()
            }

            append_dict_to_csv(f"{local_log_path}/{run_name}", metrics)

            # Log to W&B
            wandb.log({"train/loss": loss.item(), "step": step})

        step += 1

    wandb.finish()

def append_dict_to_csv(file_path, data):
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0

    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())

        # If file is empty, write header first
        if not file_exists:
            writer.writeheader()

        writer.writerow(data)