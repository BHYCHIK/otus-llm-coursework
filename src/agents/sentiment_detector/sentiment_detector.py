from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentDetector:
    def __init__(self, model_path: str):
        id2label = {0: "negative", 1: "neutral", 2: "positive"}
        label2id = {v: k for k, v in id2label.items()}
        num_labels = len(id2label)

        self._model_path = model_path
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            # ignore_mismatched_sizes=True,  # включите, если ругается на размер классификационной головы
        )

    @torch.inference_mode()
    def predict_sentiment(self, texts, max_length=256):
        self._model.eval()
        device = next(self._model.parameters()).device
        batch = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = self._model(**batch)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
        pred_ids = probs.argmax(axis=-1)
        return [
            {
                "label": self._model.config.id2label[int(i)],
                "scores": {self._model.config.id2label[j]: float(p[j]) for j in range(probs.shape[1])},
            }
            for i, p in zip(pred_ids, probs)
        ]