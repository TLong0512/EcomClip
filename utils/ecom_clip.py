import clip
import torch
import torch.nn as nn


class EcomClip(nn.Module):
    def __init__(self, clip_model_name: str = "ViT-B/32", num_classes: int = 10, device=None):
        """
        EcomClip: Wrapper for OpenAI CLIP with an additional classification head.

        :param clip_model_name: Name of the CLIP model to load (e.g., "ViT-B/32")
        :param num_classes: Number of output classes for classification
        :param device: Device to load the model onto (defaults to CUDA if available)
        """
        super(EcomClip, self).__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=self.device, jit=False)
        self.clip_model.eval()  # Freeze CLIP weights

        self.classifier = nn.Linear(self.clip_model.visual.output_dim, num_classes)

    def forward(self, images):
        with torch.no_grad():
            features = self.clip_model.encode_image(images).float()
        output = self.classifier(features)
        return output

    def get_preprocess(self):
        """
        Return the preprocessing function from CLIP.
        """
        return self.preprocess

    def get_backbone(self):
        """
        Return the CLIP model backbone (visual encoder).
        """
        return self.clip_model
