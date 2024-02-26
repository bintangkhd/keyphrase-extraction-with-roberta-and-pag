import torch
import re

import pandas as pd

class PagCentralityRanking:
    def __init__(self, input_data, noun_phrase, output_embedding, betha=0.2):
        self.keyphrase_candidate = noun_phrase[0]
        self.data = input_data
        self.betha = betha
        self.vertice = output_embedding
        self.pag_score = []
        self.traditional_score = [] 
        self.pag_keyphrase_index = []
        self.traditional_keyphrase_index = []
        self.pag_selected_keyphrase = []
        self.traditional_selected_keyphrase = []
        self.pag_all_rank_keyphrase = []
        self.traditional_all_rank_keyphrase = []

    def get_target_position_bias(self, target_word):
        separator_pattern = re.compile(r"\W+")
        document_words = separator_pattern.split(self.data.lower())
        input_words = separator_pattern.split(target_word.lower())

        word_position = 0

        for i in range(len(document_words) - len(input_words) + 1):
            if document_words[i: i + len(input_words)] == input_words:
                word_position = i
                break

        return torch.tensor([word_position + 1])

    def get_next_target_position_bias(self, target_word):
        separator_pattern = re.compile(r"\W+")
        document_words = separator_pattern.split(self.data.lower())
        input_words = separator_pattern.split(target_word.lower())

        word_position = 0

        for i in range(len(document_words) - len(input_words) + 1):
            if document_words[i: i + len(input_words)] == input_words:
                word_position += i
                break

        return torch.tensor([word_position + 1])

    def calculate_exponential_value(self, i):
        expv2_sum = torch.tensor([1.0])
        expv1 = torch.exp(torch.tensor([1]) / self.get_target_position_bias(self.keyphrase_candidate[i]))

        for idx in range(1, i + 1):
            for np in self.keyphrase_candidate[idx]:
                expv2 = torch.exp(1 / self.get_next_target_position_bias(" ".join(np)))
                expv2_sum += expv2 

        return expv1 / expv2_sum

    def traditional_centrality_scoring(self, vertice_i):
        traditional_centrality_score = torch.tensor([0.0])

        for j in range(len(self.vertice)):
            edge_ij = vertice_i * self.vertice[j]
            theta = self.betha * (torch.max(edge_ij) - torch.min(edge_ij))
            traditional_centrality_score += torch.max(edge_ij - theta)

        return traditional_centrality_score

    def pag_centrality_scoring(self):
        self.pag_score = torch.tensor([0.0] * len(self.vertice))
        self.traditional_score = torch.tensor([0.0] * len(self.vertice))

        for i in range(len(self.pag_score)):
            self.pag_score[i] = self.calculate_exponential_value(i) * self.traditional_centrality_scoring(self.vertice[i])
            self.traditional_score[i] = self.traditional_centrality_scoring(self.vertice[i])

        top_indices = torch.argsort(self.pag_score, descending=True)
        top_traditional = torch.argsort(self.traditional_score, descending=True)

        self.traditional_keyphrase_index = top_traditional.tolist()
        self.pag_keyphrase_index = top_indices.tolist()

        # Mengisi variabel pag_all_rank_keyphrase
        self.fill_pag_all_rank_keyphrase()

        # Mengisi variabel traditional_all_rank_keyphrase
        self.fill_traditional_all_rank_keyphrase()

    def fill_pag_all_rank_keyphrase(self):
        for i, index in enumerate(self.pag_keyphrase_index):
            if 0 <= index < len(self.keyphrase_candidate):
                self.pag_all_rank_keyphrase.append([i+1, self.keyphrase_candidate[index], self.pag_score[index]])

    def fill_traditional_all_rank_keyphrase(self):
        for i, index in enumerate(self.traditional_keyphrase_index):
            if 0 <= index < len(self.keyphrase_candidate):
                self.traditional_all_rank_keyphrase.append([i+1, self.keyphrase_candidate[index], self.traditional_score[index]])

    def get_pag_keyphrase_by_index(self):
        keyphrases = []
        for index in self.pag_keyphrase_index:
            if 0 <= index < len(self.keyphrase_candidate):
                keyphrases.append(self.keyphrase_candidate[index])
                self.pag_selected_keyphrase.append(self.keyphrase_candidate[index])
            else:
                keyphrases.append(None)
                self.pag_selected_keyphrase.append(None)

    def get_traditional_keyphrase_by_index(self):
        keyphrases = []
        for index in self.traditional_keyphrase_index:
            if 0 <= index < len(self.keyphrase_candidate):
                keyphrases.append(self.keyphrase_candidate[index])
                self.traditional_selected_keyphrase.append(self.keyphrase_candidate[index])
            else:
                keyphrases.append(None)
                self.traditional_selected_keyphrase.append(None)

    # def pag_score_to_csv(self):
    #     # Create DataFrame for embeddings
    #     pag_score_df = pd.DataFrame(self.pag_score.numpy())

    #     # Save DataFrame to CSV file
    #     pag_score_df.to_csv('pag_score.csv', index=False)

