import torch
from transformers import WhisperModel
from abc import ABC, abstractmethod
from ..training.MangoTrainer import TrainingOutput
from dataclasses import dataclass
from sklearn.metrics import f1_score
from ..utils.multilabel import render_confusion_matrix


class Embedder(torch.nn.Module, ABC):
    def __init__(self, model, emb_dim):
        super().__init__()
        self.model = model
        self.emb_dim = emb_dim

    @abstractmethod
    def forward(self, input: torch.Tensor, attention_mask: torch.Tensor):
        """
        :param input: shape (B x MELS x EMB)
        :param attention_mask: (B x EMB)
        :return: dict(
            embedder_output=torch tensor of shape (B x TIME x OUT),
            attention_mask: the torch.Tensor of output masked lengths (B x TIME)
        )
        """
        pass


class WhisperTinyEmbedder(Embedder):
    def __init__(self):
        whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
        super().__init__(whisper_model.encoder, 384)

    def forward(self, input: torch.Tensor, attention_mask: torch.Tensor):
        """
        Look for Embedder docs
        """
        output = self.model(input_features=input, attention_mask=attention_mask).last_hidden_state
        attention_mask = attention_mask[:, ::2]
        return {
            "embedder_output": output,
            "attention_mask": attention_mask
        }


class NoiseLinearHead(torch.nn.Module):
    def __init__(self, emb_dim, num_cl, multi_classes: bool):
        """
        :param emb_dim: the embedded dimension of each timestamp in embedded model
        :param num_cl: number of noise classes
        :param multi_classes: if True calculates sigmoid for each class, otherwise calculates clas probs
        """
        super().__init__()
        self.linear = torch.nn.Linear(emb_dim, num_cl)
        self.multi_classes = multi_classes

    def forward(self, embedder_output: torch.Tensor, attention_mask: torch.Tensor):
        """
        :params *: see the output of Embedder
        :return: tensor with sigmoids for classes (B x NUM_CL)
        """
        lengths = torch.tensor([attention_mask[i].sum() for i in range(attention_mask.shape[0])]).to(
            embedder_output.device)
        mask = torch.stack([attention_mask for _ in range(embedder_output.shape[-1])], dim=2)
        output = mask * embedder_output
        output = torch.sum(output, dim=1)
        output /= torch.stack([lengths for _ in range(output.shape[1])], dim=1)
        output = self.linear(output)
        return output


class NoiseClassifier(torch.nn.Module):
    def __init__(self, embedder: Embedder, num_noises, multi_classes=True):
        """
        :param embedder: backbone embedder
        :param num_noises: number of noise classes
        :param multi_classes: if True calculates sigmoid for each class, otherwise calculates clas probs
        """
        super().__init__()
        self.embedder = embedder
        self.head = NoiseLinearHead(embedder.emb_dim, num_noises, multi_classes)
        self.loss_fn = torch.nn.BCEWithLogitsLoss() if multi_classes else torch.nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, labels: torch.tensor, attention_mask: torch.Tensor):
        """
        :param input: shape (B x MELS x FEAT)
        :param labels: shape (B x NUM_CL)
        :param attention_mask: (B x FEAT)
        :return: dict(ce_loss=average of binary cross entropy losses, predictions=noise class predictions)
        """
        embedder_output = self.embedder(input, attention_mask)
        predictions = self.head(**embedder_output)
        loss = self.loss_fn(predictions, labels)
        return {
            "loss": loss,
            "predictions": predictions.detach(),
            "labels": labels,
        }


@dataclass
class ClassificationMulticlassAccuracy:
    threshold: float = 0.5

    def __call__(self, train_output: TrainingOutput):
        preds = train_output.model_outputs["predictions"]
        preds = (preds > self.threshold).int()
        labels = train_output.model_outputs["labels"].int()
        result = preds == labels
        accuracy = result.int().sum() / torch.numel(result)

        confusion_images = render_confusion_matrix(labels.cpu(), preds.cpu())

        return {"accuracy": float(accuracy.cpu().numpy()),
                "f1_macro": float(f1_score(y_true=labels.cpu(), y_pred=preds.cpu(), average='macro')),
                "f1_micro": float(f1_score(y_true=labels.cpu(), y_pred=preds.cpu(), average='micro')),
                "confusion_matrix": confusion_images}


class ClassificationAccuracy:
    def __call__(self, train_output: TrainingOutput):
        preds = train_output.model_outputs["predictions"]
        preds = torch.argmax(preds, dim=1)
        labels = train_output.model_outputs["labels"].int()
        labels = torch.argmax(labels, dim=1)
        result = preds == labels
        accuracy = result.int().sum() / torch.numel(result)
        return {"accuracy": float(accuracy.cpu().numpy())}