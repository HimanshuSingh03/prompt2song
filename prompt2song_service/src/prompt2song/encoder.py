from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PromptEncoder:
    def __init__(self, model_dir: Path, device: str):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {int(k): v for k, v in getattr(self.model.config, "id2label", {}).items()}

    def _tokenize_to_device(self, text: str) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def embed(self, text: str) -> np.ndarray:
        inputs = self._tokenize_to_device(text)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            token_embeddings = outputs.hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1)
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1)
            pooled = summed / counts
        return pooled[0].cpu().numpy()

    def classify(self, text: str) -> dict[str, object]:
        """Return the top predicted label and per-class probabilities for a prompt."""
        inputs = self._tokenize_to_device(text)
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
        top_idx = int(torch.argmax(probs).item())
        label = self.id2label.get(top_idx, str(top_idx))
        return {
            "label": label,
            "score": float(probs[top_idx].item()),
            "probabilities": {self.id2label.get(i, str(i)): float(p.item()) for i, p in enumerate(probs)},
        }
