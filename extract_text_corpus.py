import os
import lzma # Handle xz files
from tqdm import tqdm # Show progess bar

def xz_files_in_dir(directory):
	files = []
	for filename in os.listdir(directory):
		if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
			files.append(filename)
	return files

folder_path = "/Users/matthijskralt/code/TravelingTice/misc/my_first_llm/openwebtext"
output_file_train = "train_data/train_split.txt"
output_file_val = "train_data/val_split.txt"
vocab_file = "train_data/vocab.txt"

files = xz_files_in_dir(folder_path)
total_files = len(files)

split_index = int(total_files * 0.9) # 90% for training
files_train = files[:split_index]
files_val = files[split_index:]

vocab = set()

def process_file(files_array, target_file_name):
	with open(target_file_name, "w", encoding="utf-8") as outfile:

		for filename in tqdm(files_array, total=len(files_array)):
			file_path = os.path.join(folder_path, filename)

			with lzma.open(file_path, "rt", encoding="utf-8") as infile:
				text = infile.read()
				outfile.write(text)
				characters = set(text)
				vocab.update(characters)


process_file(files_train, output_file_train)
process_file(files_val, output_file_val)

with open(vocab_file, "w", encoding="utf-8") as vocabfile:
	for char in vocab:
		vocabfile.write(char + "\n")
