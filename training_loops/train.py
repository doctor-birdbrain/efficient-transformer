import os

import csv
import torch
from torch.utils.data import DataLoader
import wandb


# We use "_v0" to denote the latest and greatest version of the training loop
def train_loop_v0(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    aux_weight=0.5,
    device="cuda",
    project_name="vit-tiny-early-exit",
):
    # This is a decent idea if we are finetuning the entire transformer:
    lr = 1e-4
    weight_decay = 0.05
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # If we only train some head, better to use lr=1e-3, no decay, and just Adam

    wandb.init(
        project=project_name,
        config={
            "optimizer": "AdamW",
            "lr": lr,
            "weight_decay": weight_decay,
            "aux_weight": aux_weight,
            "epochs": num_epochs,
        },
    )

    for epoch in range(num_epochs):
        train_loss, train_metrics = model.train_one_epoch(
            train_loader, optimizer, aux_weight, device
        )

        train_loss, train_metrics = model.train_one_epoch(
            train_loader, optimizer, aux_weight, device
        )
        val_loss, val_metrics = model.evaluate(val_loader, aux_weight, device)

        # Log to console
        print(f"\nEpoch {epoch}")
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        for head in train_metrics:
            print(
                f"\t{head}: train={train_metrics[head]:.3f}, val={val_metrics[head]:.3f}"
            )
        # Log to wandb
        log_dict = {
            "train/loss": train_loss,
            "val/loss": val_loss,
        }
        for head, acc in train_metrics.items():
            log_dict[f"train/{head}_acc"] = acc
        for head, acc in val_metrics.items():
            log_dict[f"val/{head}_acc"] = acc

        wandb.log(log_dict)

    wandb.finish()


def train_loop(
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
    run_name="trial_run",
):
    """
    Train a PyTorch model for a fixed number of iterations and log the training loss to W&B.
    """

    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    # Create data loader
    dataloader = DataLoader(train_dataset, batch_size=batch_size)
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Simple optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # W&B init
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "batch_size": batch_size,
            "learning_rate": lr,
            "iterations": num_steps,
        },
    )

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
        if val_dataset and (
            (step + 1) % val_interval == 0 or step == num_steps - 1
        ):
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs, val_targets = (
                        val_batch["image"],
                        val_batch["label"],
                    )
                    val_inputs, val_targets = val_inputs.to(
                        device
                    ), val_targets.to(device)
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
                "train/loss": loss.item(),
            }
            wandb.log(metrics)
            append_dict_to_csv(f"{local_log_path}/{run_name}_val", metrics)

            model.train()
        else:
            metrics = {"step": step, "loss": loss.item()}

            append_dict_to_csv(f"{local_log_path}/{run_name}", metrics)

            # Log to W&B
            wandb.log({"train/loss": loss.item(), "step": step})

        step += 1

    wandb.finish()


def append_dict_to_csv(file_path, data):
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0

    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())

        # If file is empty, write header first
        if not file_exists:
            writer.writeheader()

        writer.writerow(data)
