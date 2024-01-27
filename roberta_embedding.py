from transformers import AutoModel, AutoTokenizer
import torch

class RobertaEmbedding:
    def __init__(self, noun_phrases, tensor_length=512):
        self.roberta_plm = "jafarabdurrohman/indonesian-roberta-base-ler"
        self.model_embedding = AutoModel.from_pretrained(self.roberta_plm)
        self.model_tokenizer = AutoTokenizer.from_pretrained(self.roberta_plm)
        self.noun_phrases = noun_phrases
        self.tensor_length = tensor_length
        self.output_embedding = []

    def begin_embedding(self):
        noun_phrase = [np for sublist in self.noun_phrases for np in sublist]

        token_embedding = self.model_tokenizer(
            noun_phrase,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.tensor_length,
        )

        with torch.no_grad():
            output = self.model_embedding(**token_embedding)

        avg_embedding = output.last_hidden_state.mean(dim=1)
        self.output_embedding.append(avg_embedding)