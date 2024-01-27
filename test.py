from preprocessing import Preprocessing

# Ambil input dari keyboard
judul_dokumen = input("Masukkan judul dokumen: ")
abstrak_dokumen = input("Masukkan abstrak dokumen: ")

# Uji kelas Preprocessing
preprocessor = Preprocessing(judul_dokumen, abstrak_dokumen)
preprocessor.combine_data()
preprocessor.cleansing_data()
preprocessor.pos_tagging_token()
preprocessor.noun_phrase_chunking()

# Tampilkan hasil
print("Data Input:", preprocessor.data_input)
print("\n\nToken setelah cleansing:", preprocessor.token)
print("\n\nPOS Tagged Token:", preprocessor.pos_tagged_token)
print("\n\nNoun Phrases:", preprocessor.noun_phrase)