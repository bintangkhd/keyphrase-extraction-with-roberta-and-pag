import torch
import re

class PagCentralityRanking:
    def __init__(self, input_data, noun_phrase, output_embedding, betha = 0.2):
        self.keyphrase_candidate = noun_phrase[0]
        self.data = input_data
        self.betha = betha
        self.vertice = output_embedding
        self.pag_keyphrase_index = []
        self.pag_selected_keyphrase = []
        self.traditional_keyphrase_index = []
        self.traditional_selected_keyphrase = []
        
    def get_target_position_bias(self, target_word):
        separator_pattern = re.compile(r"\W+")  # Pattern untuk memisahkan kata non-huruf
        document_words = separator_pattern.split(
            self.data.lower()
        )  # Pisahkan kata dalam dokumen dan ubah menjadi huruf kecil
        input_words = separator_pattern.split(
            target_word.lower()
        )  # Pisahkan kata dalam input dan ubah menjadi huruf kecil

        word_position = 0

        # Cari posisi teks input dalam dokumen
        for i in range(len(document_words) - len(input_words) + 1):
            if document_words[i: i + len(input_words)] == input_words:
                word_position = i
                break

        return torch.tensor([word_position + 1])
    
    def get_next_target_position_bias(self, target_word):
        separator_pattern = re.compile(r"\W+")  # Pattern untuk memisahkan kata non-huruf
        document_words = separator_pattern.split(
            self.data.lower()
        )  # Pisahkan kata dalam dokumen dan ubah menjadi huruf kecil
        input_words = separator_pattern.split(
            target_word.lower()
        )  # Pisahkan kata dalam input dan ubah menjadi huruf kecil

        word_position = 0

        # Cari posisi teks input dalam dokumen
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
        pag_score = torch.tensor([0] * len(self.vertice))
        traditional_score = torch.tensor([0] * len(self.vertice))

        for i in range(len(pag_score)):
            pag_score[i] = self.calculate_exponential_value(i) * self.traditional_centrality_scoring(self.vertice[i])
            traditional_score[i] = self.traditional_centrality_scoring(self.vertice[i])
            
        top_indices = torch.argsort(pag_score, descending=True)[:5]  # Mendapatkan 5 indeks terbesar
        top_traditional = torch.argsort(traditional_score, descending=True)[:5]

        self.traditional_keyphrase_index = top_traditional.tolist()
        self.pag_keyphrase_index = top_indices.tolist()
    
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
                self.traditional_selected_keyphrase.append(
                    self.keyphrase_candidate[index])
            else:
                keyphrases.append(None)
                self.traditional_selected_keyphrase.append(None)