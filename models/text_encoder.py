import clip
import torch


class TextEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/32", device=self.device)

    def encode(self, text):
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {t}") for t in text])
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs.to(self.device))
        return text_features