import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def sparse_channel_pruning(model, pruning_percentage=0.2):
    """
    Performs sparse channel pruning on an image classification model.

    Args:
        model (nn.Module): The image classification model to prune.
        pruning_percentage (float): The percentage of channels to prune (0.0 to 1.0).

    Returns:
        nn.Module: The pruned model.
    """

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=pruning_percentage, n=1, dim=0)
            #Older pytorch compatibility
            for buffer_name in list(module.named_buffers()):
                if "weight_mask" in buffer_name or "weight_orig" in buffer_name:
                    delattr(module, buffer_name[0])

    return model


def save_pruned_model(model, feature_extractor, output_dir):
    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    print(f"Pruned model saved to {output_dir}")