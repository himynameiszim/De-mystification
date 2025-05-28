import os
import re

def split_text_into_sentences(text: str) -> list[str]:
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]

def read_txt_files_to_sentences_dict(folder_path: str) -> dict:
    """
    Reads all .txt files in a given folder and splits their content into sentences.
    :param folder_path: Path to the folder containing .txt files (presumable corpora). 
    :return sentences_dict: a dictionary where the keys values are filenames and the values are the lists of sentences
                            in the corresponding files.
    """
    sentences_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                sentences = split_text_into_sentences(text)
                # Use filename without extension as key, or keep full filename as key
                key = os.path.splitext(filename)[0]
                sentences_dict[key] = sentences
    return sentences_dict