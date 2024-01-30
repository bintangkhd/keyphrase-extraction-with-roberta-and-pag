import streamlit as st
from preprocessing import Preprocessing
from roberta_embedding import RobertaEmbedding
from pag_centrality_ranking import PagCentralityRanking

class Main:
    def __init__(self):
        self.input_title = " "
        self.input_abstract = " "
        self.golden_keyphrase = []
        self.confusion_matrix = []
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0

    def input_gui(self):
        self.input_title = st.text_area("Masukkan Judul Dokumen:")
        self.input_abstract = st.text_area("Masukkan Abstrak Dokumen:")
        self.golden_keyphrase = st.text_area("Masukkan Golden keyphrase :")
        self.golden_keyphrase = self.golden_keyphrase.split(";")

    def confusion_matrix_evaluation(self, candidate_keyphrase, selected_keyphrase, golden_keyphrase):
        TP = TN = FP = FN = 0

        for item in selected_keyphrase:
            if item in golden_keyphrase:
                TP += 1
            else:
                FP += 1

        for item in candidate_keyphrase:
            if item not in selected_keyphrase and item not in golden_keyphrase:
                TN += 1

        for item in golden_keyphrase:
            if item not in selected_keyphrase:
                FN += 1

        self.precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        self.recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        # self.accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
        self.f1_score = (2 * (self.precision * self.recall) / (self.precision + self.recall)
            if (self.precision + self.recall) != 0 else 0
        )
        
        st.write("True Positives (TP):", TP)
        st.write("True Negatives (TN):", TN)
        st.write("False Positives (FP):", FP)
        st.write("False Negatives (FN):", FN)
        st.write("Precision: ", self.precision)
        st.write("Recall:", self.recall)
        # st.write("Accuracy:", self.accuracy)
        st.write("F1-Score:", self.f1_score)


main = Main()
input_gui = main.input_gui()

preprocessing = Preprocessing(main.input_title, main.input_abstract)

if st.button("Generate Keyphrase"):
    if main.input_title and main.input_abstract:
        preprocessing = Preprocessing(main.input_title, main.input_abstract)
        preprocessing.combine_data()
        preprocessing.cleansing_data()
        preprocessing.pos_tagging_token()
        preprocessing.noun_phrase_chunking()

        # st.write('token + stop word :',preprocessor.data)
        st.write("Gabungan Judul dan Abstrak:", preprocessing.data_input)
        st.write("Teks yang Dibersihkan:", preprocessing.token)
        st.write("POS Tagged Tokens:", preprocessing.pos_tagged_token)
        st.write("Noun Phrases:", preprocessing.noun_phrase)
        print("Noun Phrases:", preprocessing.noun_phrase[0])

        roberta_embedding = RobertaEmbedding(preprocessing.noun_phrase)
        roberta_embedding.begin_embedding()
        st.write('tipe embedding : ', type(roberta_embedding.output_embedding))
        st.write("Hasil Embedding : ", roberta_embedding.output_embedding)

        pag_ranking = PagCentralityRanking(preprocessing.data_input, preprocessing.noun_phrase, roberta_embedding.output_embedding[0])
        pag_ranking.pag_centrality_scoring()
        pag_ranking.get_pag_keyphrase_by_index()
        pag_ranking.get_traditional_keyphrase_by_index()

        st.write("Golden Keyphrase : ", main.golden_keyphrase)

        st.write("Selected Keyphrase by PAG: ", pag_ranking.pag_selected_keyphrase)
        print("Selected Keyphrase by PAG: ", pag_ranking.pag_selected_keyphrase)

        st.write('Evaluasi Confusion Matrix PAG')
        main.confusion_matrix_evaluation(preprocessing.noun_phrase[0], pag_ranking.pag_selected_keyphrase, main.golden_keyphrase)

        st.write("Selected Keyphrase by Traditional Centrality : ", pag_ranking.traditional_selected_keyphrase)
        print("Selected Keyphrase by Traditional Centrality : ", pag_ranking.traditional_selected_keyphrase)

        st.write('Evaluasi Confusion Matrix Traditional Centrality')
        main.confusion_matrix_evaluation(preprocessing.noun_phrase[0], pag_ranking.traditional_selected_keyphrase, main.golden_keyphrase)

    else:
        st.warning("Judul dan Abstrak tidak boleh kosong!")