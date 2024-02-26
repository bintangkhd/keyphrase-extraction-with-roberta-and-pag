from transformers import AutoModel, AutoTokenizer
import torch

import pandas as pd

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


    # def to_csv(self):
    #     # Create DataFrame for noun phrases
    #     noun_phrase_df = pd.DataFrame({'Noun Phrase': [' '.join(phrase.split()) for sublist in self.noun_phrases for phrase in sublist]})

    #     # Create DataFrame for embeddings
    #     embedding_df = pd.DataFrame(self.output_embedding[0].numpy())

    #     # Combine noun phrases and embedding DataFrames
    #     combined_df = pd.concat([noun_phrase_df, embedding_df], axis=1)

    #     # Save DataFrame to CSV file
    #     combined_df.to_csv('average_embedding.csv', index=False)