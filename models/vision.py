import collections
import torch.nn as nn
import torch.nn.functional as F


class ViTWithAuxHeads(nn.Module):
    def __init__(
        self, model_name, num_classes, aux_layers=[3, 6, 9], pretrained=True
    ):
        super.__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.embed_dim = base_model.embed_dim

        # TODO: swap out other parts?
        self.vit.head = nn.Linear(self.embed_dim, num_classes)
        self.aux_layers = aux_layers
        self.aux_heads = nn.ModuleDict()

        for layer in self.aux_layers:
            vit.aux_heads[str(layer)] = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(drop_rate),
                nn.Linear(self.embed_dim, num_classes),
            )

    def forward_with_aux(self, x):
        """
        We follow the form of the forward function of timm's ViT,
        specifically the "forward_features" function.
        https://github.com/huggingface/pytorch-image-models/blob/954613a470652e4a113ff45b62dbd15c4e229218/timm/models/vision_transformer.py#L934C9-L934C25
        """
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        logits_dict = {}
        for i, block in enumerate(self.vit.blocks, start=1):
            x = block(x)
            if i in self.aux_layers:
                cls_rep = x[:, 0]  # the quirky CLS token
                logits_dict[str(i)] = self.aux_heads[str(i)](cls_rep)

        logits_dict["final"] = self.vit.forward_head(x)

        return logits_dict

    def compute_loss(self, logits_dict, targets, aux_weight=0.5):
        """
        Computes loss.
        """

        aux_loss = 0.0
        for k, logits in logits_dict.items():
            if k == "final":
                main_loss = F.cross_entropy(logits, targets)
            else:
                aux_loss += F.cross_entropy(logits, targets)

        if len(self.aux_layers) > 0:
            loss = (
                self.aux_weight * (aux_loss / len(self.aux_layers))
                + (1 - self.aux_weight) * main_loss
            )
        else:
            loss = main_loss

        return loss

    @torch.no_grad()
    def predict_with_early_exit(x, threshold=0.9, exit_logging=True):
        """
        Takes a timm ViT model and runs inference with early exits.

        We follow the form of the forward function of timm's ViT,
        specifically the "forward_features" function.
        https://github.com/huggingface/pytorch-image-models/blob/954613a470652e4a113ff45b62dbd15c4e229218/timm/models/vision_transformer.py#L934C9-L934C25
        """

        # Outputs to fill (maintain original batch order)
        preds = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
        confs = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
        if exit_logging:
            exit_at = [None] * x.shape[0]
        else:
            exit_at = None

        # Common initial part, runs on full batch
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        # Keep track of inputs still active
        active_idx = torch.arange(x.shape[0], device=x.device)

        # We skip the case of attention masking, not needed for
        # currently planned experiments.
        for i, block in enumerate(self.vit.blocks, start=1):
            if active_idx.numel() == 0:
                break  # all batch items have exited

            x = block(x)

            if i in exit_layers:
                # It seems we need to do this because it's the only
                # available representation at intermediate depths
                cls_rep = x[:, 0]  # the quirky CLS token
                logits = self.vit.aux_heads[str(i)](cls_rep)
                probs = F.softmax(logits, dim=-1)
                conf, pred = probs.max(dim=-1)

                # Decide which elements exit
                exit_mask = conf >= threshold
                if exit_mask.any():
                    exited_orig_idx = active_idx[exit_mask]
                    preds[exited_orig_idx] = pred[exit_mask]
                    confs[exited_orig_idx] = conf[exit_mask]
                    if exit_logging:
                        for j in exited_orig_idx.tolist():
                            exit_at[j] = i

                    # Keep only the not-exited samples for deeper blocks
                    keep_mask = ~exit_mask
                    # Shrink batch:
                    x = x[keep_mask]
                    # Shrink index mapping:
                    active_idx = active_idx[keep_mask]

        # If we didn't exit early...
        if active_idx.numel() > 0:
            logits = self.vit.forward_head(x)
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)

            preds[active_idx] = pred
            confs[active_idx] = conf
            if exit_logging:
                for j in active_idx.tolist():
                    exit_at[j] = "final"

        return preds, exit_at, confs

    def accuracy(self, logits, targets):
        preds = logits.argmax(dim=-1)
        return (preds == targets).float().mean().item()

    def evaluate(self, val_dataloader, aux_weight=0.5, device="cuda"):
        self.eval()
        total_loss = 0.0
        total_metrics = collections.defaultdict(list)

        batch_count = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                batch_count += 1
                images, labels = images.to(device), labels.to(device)

                logits_dict = self.forward_with_aux(images)
                loss = self.compute_loss(
                    logits_dict, labels, aux_weight=aux_weight
                )
                total_loss += loss.item()

                for head_name, logits in logits_dict.items():
                    acc = accuracy(logits, labels)
                    total_metrics[head_name].append(acc)

        avg_loss = total_loss / batch_count
        avg_metrics = {k: sum(v) / len(v) for k, v in total_metrics.items()}
        return avg_loss, avg_metrics

    def evaluate_early_exit(
        self,
        val_dataloader,
        threshold=0.9,
        device="cuda",
        log_to_wandb=False,
        epoch=None,
    ):
        """
        Evaluation assuming the model can do early exits at eval time.
        """
        self.eval()
        correct, total = 0, 0

        # track stats per exit
        exit_correct, exit_total = {}, {}

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)

                # Predict with early exit
                preds, exit_at, _ = self.predict_with_early_exit(
                    images, threshold=threshold
                )

                total += targets.size(0)
                correct += (preds == targets).sum().item()

                # per-exit stats
                for i, ex in enumerate(exit_at):
                    exit_total[ex] = exit_total.get(ex, 0) + 1
                    exit_correct[ex] = (
                        exit_correct.get(ex, 0)
                        + (preds[i] == targets[i]).item()
                    )

        # aggregate results
        overall_acc = correct / total
        per_exit_acc = {
            ex: exit_correct[ex] / exit_total[ex] for ex in exit_total
        }

        print(f"[Validation] Overall Acc: {overall_acc:.4f}")
        for ex, acc in per_exit_acc.items():
            print(f"  Exit {ex}: {acc:.4f} (n={exit_total[ex]})")

        # log to wandb if requested
        if log_to_wandb:
            log_dict = {
                f"val/exit_{ex}_acc": acc for ex, acc in per_exit_acc.items()
            }
            log_dict["val/overall_acc"] = overall_acc
            log_dict["epoch"] = epoch

            import wandb

            wandb.log(log_dict)

        return overall_acc, per_exit_acc

    def train_one_epoch(
        self,
        train_dataloader,
        optimizer,
        aux_weight,
        device="cuda",
    ):
        self.train()
        total_loss = 0.0
        total_metrics = collections.defaultdict(list)

        batch_count = 0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            logits_dict = self.forward_with_aux(images)
            loss = self.compute_loss(
                logits_dict, labels, aux_weight=aux_weight
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

            # Track per-head accuracy
            with torch.no_grad():
                for head_name, logits in logits_dict.items():
                    acc = self.accuracy(logits, labels)
                    total_metrics[head_name].append(acc)

        avg_loss = total_loss / batch_count
        avg_metrics = {k: sum(v) / len(v) for k, v in total_metrics.items()}
        return avg_losses, avg_metrics
