import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy

import evaluate
import wandb

from tqdm.auto import tqdm


def do_evaluate_fancy(model, test_dataset, batch_size=8, project_name="dummy-eval"):

    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model.eval()

    wandb.init(project=project_name, name="test-run", mode="online")

    accuracy_metric = evaluate.load("accuracy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)
    
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
    
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_metric.compute(predictions=all_preds, references=all_labels)

    wandb.log({"accuracy": acc["accuracy"]})

    print(f"Accuracy: {acc['accuracy']:.4f}")
    wandb.finish()

def do_evaluate_simple(model, test_dataset, batch_size=8):
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model.eval()

    metric = MulticlassAccuracy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)
    
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
    
            metric.update(preds, y)

    accuracy = metric.compute()
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

