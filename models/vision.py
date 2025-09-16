import collections

from fvcore.nn import FlopCountAnalysis
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


# Example of the path string
def load_from_wandb(
    model_name, wandb_string, num_classes, aux_layers=[3, 6, 9]
):
    artifact = wandb.use_artifact(wandb_string, type="model")
    artifact_dir = artifact.download()
    state_dict = torch.load(f"{artifact_dir}/vit_with_auxheads.pth")

    model = VitWithAuxHeads(
        model_name, num_classes, aux_layers, pretrained=False
    )
    model.load_state_dict(state_dict)
    return model


class ViTWithAuxHeads(nn.Module):
    def __init__(
        self, model_name, num_classes, aux_layers=[3, 6, 9], pretrained=True
    ):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.embed_dim = self.vit.embed_dim

        # TODO: swap out other parts?
        self.vit.head = nn.Linear(self.embed_dim, num_classes)
        self.aux_layers = aux_layers
        self.aux_heads = nn.ModuleDict()

        for layer in self.aux_layers:
            self.aux_heads[str(layer)] = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(self.vit.head_drop.p),
                nn.Linear(self.embed_dim, num_classes),
            )

        # -- Next, we pre-compute how many flops for each block and heads.
        # We will use this at eval time to compute the cost of inference
        self.block_flops = {}
        self.head_flops = {}

        # We need to find the sequence length of this model
        # It will be num_patches + cls_token
        # 1. Number of patches the model creates
        num_patches = self.vit.patch_embed.num_patches
        # 2. Add 1 if model uses a class token
        seq_len = num_patches + (
            1
            if hasattr(self.vit, "cls_token")
            and self.vit.cls_token is not None
            else 0
        )

        for i, block in enumerate(self.vit.blocks, start=1):
            fa = FlopCountAnalysis(
                block, torch.randn(1, seq_len, self.vit.embed_dim)
            )
            self.block_flops[i] = fa.total()

        for layer in self.aux_layers:
            # The heads don't need seq_len, because the head
            # only takes the CLS token as input.
            # The flops from this head
            fa = FlopCountAnalysis(
                self.aux_heads[str(layer)],
                torch.randn(1, base_model.embed_dim),
            )
            # The flops from layers leading up to this head
            block_flops = sum(
                [self.block_flops[i] for i in range(1, layer + 1)]
            )
            self.head_flops[str(layer)] = fa.total() + block_flops

        # And now the final head
        fa = FlopCountAnalysis(block, torch.randn(1, base_model.embed_dim))
        self.head_flops["final"] = fa.total()

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
                aux_weight * (aux_loss / len(self.aux_layers))
                + (1 - aux_weight) * main_loss
            )
        else:
            loss = main_loss

        return loss

    @torch.no_grad()
    def predict_with_early_exit(self, x, threshold=0.9, exit_logging=True):
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

            if i in self.aux_layers:
                # It seems we need to do this because it's the only
                # available representation at intermediate depths
                cls_rep = x[:, 0]  # the quirky CLS token
                logits = self.aux_heads[str(i)](cls_rep)
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
            for batch in val_dataloader:
                batch_count += 1
                images, labels = batch["image"].to(device), batch["label"].to(
                    device
                )

                logits_dict = self.forward_with_aux(images)
                loss = self.compute_loss(
                    logits_dict, labels, aux_weight=aux_weight
                )
                total_loss += loss.item()

                for head_name, logits in logits_dict.items():
                    acc = self.accuracy(logits, labels)
                    total_metrics[head_name].append(acc)

        avg_loss = total_loss / batch_count
        avg_metrics = {k: sum(v) / len(v) for k, v in total_metrics.items()}
        return avg_loss, avg_metrics

    def evaluate_early_exit(
        self,
        val_dataloader,
        aux_weight,
        threshold=0.9,
        log_to_wandb=True,
        epoch=None,
        device="cuda",
    ):
        """
        Evaluation assuming the model can do early exits at eval time.
        """
        self.eval()
        correct, total = 0, 0
        total_loss = 0.0
        batch_count = 0

        flop_count = 0

        # track stats per exit
        exit_correct, exit_total = {}, {}

        with torch.no_grad():
            for batch in val_dataloader:
                batch_count += 1
                images, labels = batch["image"].to(device), batch["label"].to(
                    device
                )

                # To calculate the validation loss
                logits_dict = self.forward_with_aux(images)
                loss = self.compute_loss(
                    logits_dict, labels, aux_weight=aux_weight
                )
                total_loss += loss.item()

                # To get the accuracy per exit: predict with early exit
                preds, exit_at, _ = self.predict_with_early_exit(
                    x=images, threshold=threshold, exit_logging=True
                )
                total += labels.size(0)
                correct += (preds == labels).sum().item()

                # per-exit stats
                for i, ex in enumerate(exit_at):
                    exit_total[ex] = exit_total.get(ex, 0) + 1
                    exit_correct[ex] = (
                        exit_correct.get(ex, 0)
                        + (preds[i] == labels[i]).item()
                    )
                    flop_count += self.head_flops[ex]

        # aggregate results
        overall_acc = correct / total
        per_exit_acc = {
            ex: exit_correct[ex] / exit_total[ex] for ex in exit_total
        }
        val_loss = total_loss / batch_count

        print(f"[Validation] FLOPs: {flop_count}")
        print(f"[Validation] Overall Acc: {overall_acc:.4f}")
        for ex, acc in per_exit_acc.items():
            print(f"  Exit {ex}: {acc:.4f} (n={exit_total[ex]})")

        # log to wandb if requested
        if log_to_wandb:
            log_dict = {
                f"val/exit_{ex}_acc": acc for ex, acc in per_exit_acc.items()
            }
            log_dict["val/overall_acc"] = overall_acc
            log_dict["val/flops"] = flop_count
            log_dict["epoch"] = epoch

            wandb.log(log_dict)

        return overall_acc, per_exit_acc, val_loss, flop_count

    def train_one_epoch(
        self,
        train_dataloader,
        optimizer,
        aux_weight,
        confidence_threshold,
        device="cuda",
    ):
        self.train()
        total_loss = 0.0
        total_metrics = collections.defaultdict(list)

        batch_count = 0
        for batch in train_dataloader:
            images, labels = batch["image"].to(device), batch["label"].to(
                device
            )
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
        return avg_loss, avg_metrics
