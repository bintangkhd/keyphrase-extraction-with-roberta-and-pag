import streamlit as st
import pandas as pd
from preprocessing import Preprocessing
from roberta_embedding import RobertaEmbedding
from pag_centrality_ranking import PagCentralityRanking

st.title('_Keyphrase Extraction_ Menggunakan :blue[RoBERTa dan _Position Aware Graph_]')

class Gui:
    def __init__(self):
        self.input_title = ""
        self.input_abstract = ""
        self.pag_selected_keyphrase = ""
        self.traditional_selected_keyphrase = ""
        self.golden_keyphrase = ""

    def input_gui(self):
        self.input_title = st.text_area("Judul Dokumen:", placeholder="Masukkan judul dokumen")
        self.input_abstract = st.text_area("Abstrak Dokumen:", height=350, placeholder="Masukkan abstrak dokumen")

    def extract_keyphrase(self):
        if st.button("Ekstrak _Keyphrase_"):
            if self.input_title and self.input_abstract:
                preprocessing = Preprocessing(self.input_title, self.input_abstract)
                preprocessing.combine_data()
                preprocessing.cleansing_data()
                preprocessing.pos_tagging_token()
                preprocessing.noun_phrase_chunking()

                st.session_state.candidate_keyphrase = preprocessing.noun_phrase

                roberta_embedding = RobertaEmbedding(preprocessing.noun_phrase)
                roberta_embedding.begin_embedding()
                st.session_state.output_embedding = roberta_embedding.output_embedding

                pag_ranking = PagCentralityRanking(preprocessing.data_input, preprocessing.noun_phrase, roberta_embedding.output_embedding[0])
                pag_ranking.pag_centrality_scoring()
                pag_ranking.get_pag_keyphrase_by_index()
                pag_ranking.get_traditional_keyphrase_by_index()

                self.pag_selected_keyphrase = pag_ranking.pag_selected_keyphrase
                self.traditional_selected_keyphrase = pag_ranking.traditional_selected_keyphrase
                
                st.session_state.extraction_done = True
                st.session_state.pag_selected_keyphrase = self.pag_selected_keyphrase
                st.session_state.traditional_selected_keyphrase = self.traditional_selected_keyphrase
            else:
                st.warning("Judul dan abstrak dokumen tidak boleh kosong!")

    def display_selected_keyphrases(self):
        st.write("---")
        st.subheader("Keyphrase Terpilih")
        col1, col2 = st.columns(2)

        with col1:
            self.pag_selected_keyphrase = st.session_state.get("pag_selected_keyphrase", [])
            st.text_area("_Position Aware Graph_", value="\n".join(self.pag_selected_keyphrase), height=150, key="pag_keyphrase_extract")

        with col2:
            self.traditional_selected_keyphrase = st.session_state.get("traditional_selected_keyphrase", [])
            st.text_area("_Traditional Centrality_", value="\n".join(self.traditional_selected_keyphrase), height=150, key="traditional_keyphrase_extract")

    def action_buttons(self):
        col1, col2, col3, col4 = st.columns([3, 3, 4, 3])
        with col1:
            if st.button("Lihat Pembahasan", key="show_explanation_button", disabled=not st.session_state.get("extraction_done", False)):
                    st.session_state.page_state = "explanation"
        with col2:
            if st.button("Evaluasi Hasil", disabled=not st.session_state.get("extraction_done", False)):
                st.session_state.page_state = "evaluation"
        with col3:
            pass
        with col4:
            if st.button("Mulai Ulang Sesi", key="reload_session"):
                st.session_state.clear()
                st.rerun()
        
        st.write("---")

    def show_explanation(self):
        if st.session_state.get("page_state") == "explanation":
            st.header("Pembahasan")
            preprocessing = Preprocessing(self.input_title, self.input_abstract)
            preprocessing.combine_data()
            preprocessing.cleansing_data()
            preprocessing.pos_tagging_token()
            preprocessing.noun_phrase_chunking()

            st.write("Data Input: ")
            st.write(preprocessing.data_input)
            st.write("Teks yang Dibersihkan:", preprocessing.token)
            st.write("_POS Tagged Tokens_:", preprocessing.pos_tagged_token)
            st.write("_Noun Phrases_:", preprocessing.noun_phrase)

            st.write("Hasil Embedding : ", st.session_state.output_embedding)
            
            st.write("_Keyphrase Terpilih_ oleh _Position Aware Graph_ : ", self.pag_selected_keyphrase)
            st.write("_Keyphrase Terpilih_ oleh _Traditional Centrality_ : ", self.traditional_selected_keyphrase)
        
    def evaluate_result(self):
        if st.session_state.get("page_state") == "evaluation":
            st.header("Evaluasi _Confusion Matrix_")
            self.golden_keyphrase = st.text_area("_Golden Keyphrase_ :", placeholder="Masukkan _golden keyphrase_ (pisahkan dengan tanda koma , )")
            self.golden_keyphrase = self.golden_keyphrase.lower()
            self.golden_keyphrase = self.golden_keyphrase.split(", ")

            btn_eval = st.button("Jalankan Evaluasi")
            if btn_eval:
                st.subheader("Evaluasi Hasil Keyphrase _Position Aware Graph_")
                self.confusion_matrix_evaluation(st.session_state.candidate_keyphrase[0], st.session_state.pag_selected_keyphrase, self.golden_keyphrase)
                st.subheader("")
                st.subheader("Evaluasi Hasil Keyphrase _Traditional Centrality_")
                self.confusion_matrix_evaluation(st.session_state.candidate_keyphrase[0], st.session_state.traditional_selected_keyphrase, self.golden_keyphrase)

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

        precision = round(TP / (TP + FP), 2) if (TP + FP) != 0 else 0
        recall = round(TP / (TP + FN), 2) if (TP + FN) != 0 else 0
        f1_score = round((2 * (precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0, 2)

        # Konversi nilai-nilai TP, TN, FP, FN menjadi string tanpa koma
        TP_str = str(int(TP))
        TN_str = str(int(TN))
        FP_str = str(int(FP))
        FN_str = str(int(FN))

        # Menampilkan tabel dengan HTML untuk mengatur penampilan teks dalam kolom
        st.markdown(
            f"""
            <table>
                <tr>
                    <th>TP</th>
                    <th>TN</th>
                    <th>FP</th>
                    <th>FN</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                </tr>
                <tr>
                    <td style="text-align: center;">{TP_str}</td>
                    <td style="text-align: center;">{TN_str}</td>
                    <td style="text-align: center;">{FP_str}</td>
                    <td style="text-align: center;">{FN_str}</td>
                    <td style="text-align: center;">{precision}</td>
                    <td style="text-align: center;">{recall}</td>
                    <td style="text-align: center;">{f1_score}</td>
                </tr>
            </table>
            """,
            unsafe_allow_html=True
        )


main = Gui()
main.input_gui()
main.extract_keyphrase()
main.display_selected_keyphrases()
main.action_buttons()
main.show_explanation()
main.evaluate_result()
