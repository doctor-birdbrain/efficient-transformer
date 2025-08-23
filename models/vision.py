import torch.nn.functional as F

def attach_aux_heads(model, exit_layers, num_classes, drop_rate=0.0):
    model.aux_layers = exit_layers
    model.aux_heads = nn.ModuleDict()

    for layer in exit_layers:
        model.aux_heads[str(layer)] = nn.Sequential(
            nn.LayerNorm(model.embed_dim),
            nn.Dropout(drop_rate),
            nn.Linear(model.embed_dim, num_classes)
        )
    return model

@torch.no_grad()
def predict_with_early_exit(model,
                            x,
                            exit_layers,
                            threshold=0.9):
    
    """
    Takes a timm ViT model and runs inference with early exits.
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
    x = model.patch_embed(x)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    x = model.norm_pre(x)

    # We skip the case of attention masking, not needed for
    # currently planned experiments.
    for i, block in enumerate(model.blocks, start=1):
        x = block(x)

        if i in exit_layers:
            # It seems we need to do this because it's the only
            # available representation at intermediate depths
            cls_rep = x[:,0] # the quirky CLS token
            logits = model.aux_heads[str(i)](cls_rep)
            probs = F.softmax(logits, dim=-1)
            confidence, prediction = probs.max(dim=-1)

            if confidence.item() >= threshold:
                return prediction.item(), i, confidence.item()

    # If we didn't exit early...
    logits = model.forward_head(x)
    probs = F.softmax(logits, dim=-1)
    confidence, prediction = probs.max(dim=-1)

    return prediction.item(), "final", confidence.item()
    