import pandas as pd
from preprocessing import Preprocessing
from roberta_embedding import RobertaEmbedding
from pag_centrality_ranking import PagCentralityRanking

# Fungsi ekstraksi keyphrase dan perhitungan matriks kebingungan
def extract_and_evaluate_keyphrase(title, abstract, golden_keyphrase):
    # Lakukan ekstraksi kata kunci
    preprocessing = Preprocessing(title, abstract)
    preprocessing.combine_data()
    preprocessing.cleansing_data()
    preprocessing.pos_tagging_token()
    preprocessing.noun_phrase_chunking()

    roberta_embedding = RobertaEmbedding(preprocessing.noun_phrase)
    roberta_embedding.begin_embedding()

    pag_ranking = PagCentralityRanking(preprocessing.data_input, preprocessing.noun_phrase, roberta_embedding.output_embedding[0])
    pag_ranking.calculate_centrality_scoring()
    pag_ranking.get_pag_keyphrase_by_index()
    pag_ranking.get_traditional_keyphrase_by_index()
    pag_ranking.fill_pag_all_rank_keyphrase()
    pag_ranking.fill_traditional_all_rank_keyphrase()

    # Hitung matriks kebingungan
    # confusion_matrix = confusion_matrix_evaluation(preprocessing.noun_phrase[0], pag_ranking.pag_selected_keyphrase[:30], golden_keyphrase)
    confusion_matrix = confusion_matrix_evaluation(preprocessing.noun_phrase[0], pag_ranking.traditional_selected_keyphrase[:30], golden_keyphrase)
    
    return {
        'TP': confusion_matrix['TP'],
        'TN': confusion_matrix['TN'],
        'FP': confusion_matrix['FP'],
        'FN': confusion_matrix['FN'],
        'Precision': confusion_matrix['precision'],
        'Recall': confusion_matrix['recall'],
        'F1-Score': confusion_matrix['f1_score']
    }

def confusion_matrix_evaluation(candidate_keyphrase, selected_keyphrase, golden_keyphrase):
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

    return {
        'TP': TP_str,
        'TN': TN_str,
        'FP': FP_str,
        'FN': FN_str,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

# Membaca file CSV yang berisi data judul dan abstrak
data = pd.read_csv('./dataset/mix_dataset.csv', header=None)

# Membaca file CSV golden_keyphrase
golden_keyphrase_data = pd.read_csv('./dataset/golden_keyphrase.csv', header=None)

# Mengubah kolom menjadi list dan membagi setiap baris berdasarkan delimiter ";"
golden_keyphrases = golden_keyphrase_data[0].str.split(';')

# Inisialisasi list untuk menyimpan golden keyphrases
golden_keyphrase_list = []

# Mengubah format data ke dalam list of lists
for keyphrase in golden_keyphrases:
    golden_keyphrase_list.append(keyphrase)

# Inisialisasi list untuk menyimpan hasil evaluasi
evaluation_results = []

# Iterasi melalui setiap baris data
for index, row in data.iterrows():
    # Ambil kolom 0 sebagai judul dan kolom 1 sebagai abstrak
    title = row[0]
    abstract = row[1]

    # Ekstrak keyphrase dan evaluasi
    result = extract_and_evaluate_keyphrase(title, abstract, golden_keyphrase_list[index])

    # Tambahkan hasil evaluasi ke dalam list
    evaluation_results.append({
        'Data uji ke-': index + 1,
        **result
    })

# Simpan hasil evaluasi ke dalam file CSV
evaluation_df = pd.DataFrame(evaluation_results)
evaluation_df.to_csv('evaluation_all_data_results.csv', index=False)
