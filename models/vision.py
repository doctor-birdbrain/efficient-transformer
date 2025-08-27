import collections
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

    def forward(self, x):
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
    def predict_with_early_exit(x, threshold=0.9):
        """
        Takes a timm ViT model and rRuns inference with early exits.
        - exit layers: list of layer indices where aux heads are attached.
        - threshold: confidence threshold for early exit

        Returns:
        - predicted class
        - exit layer, or "final"
        - confidence score

        We follow the form of the forward function of timm's ViT,
        specifically the "forward_features" function.
        https://github.com/huggingface/pytorch-image-models/blob/954613a470652e4a113ff45b62dbd15c4e229218/timm/models/vision_transformer.py#L934C9-L934C25
        """
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        # We skip the case of attention masking, not needed for
        # currently planned experiments.
        for i, block in enumerate(self.vit.blocks, start=1):
            x = block(x)

            if i in exit_layers:
                # It seems we need to do this because it's the only
                # available representation at intermediate depths
                cls_rep = x[:, 0]  # the quirky CLS token
                logits = self.vit.aux_heads[str(i)](cls_rep)
                probs = F.softmax(logits, dim=-1)
                confidence, prediction = probs.max(dim=-1)

                if confidence.item() >= threshold:
                    return prediction.item(), i, confidence.item()

        # If we didn't exit early...
        logits = self.vit.forward_head(x)
        probs = F.softmax(logits, dim=-1)
        confidence, prediction = probs.max(dim=-1)

        return prediction.item(), "final", confidence.item()

    def accuracy(self, logits, targets):
        preds = logits.argmax(dim=-1)
        return (preds == targets).float().mean().item()

    def train_one_epoch(
        self,
        train_dataloader,
        optimizer,
        aux_weight,
        device="cpu",
    ):
        self.train()
        total_loss = 0.0
        total_metrics = collections.defaultdict(list)

        batch_count = 0
        for images, labels in train_dataloader:
            batch_count += 1
            logits_dict = self.forward(images)
            loss = compute_loss(logits_dict, labels, aux_weight=aux_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Track per-head accuracy
            with torch.no_grad():
                for head_name, logits in logits_dict.items():
                    acc = self.accuracy(logits, labels)
                    total_metrics[head].append(acc)

        avg_loss = total_loss / batch_count
        avg_metrics = {k: sum(v) / len(v) for k, v in total_metrics.items()}
        return avg_losses, avg_metrics
