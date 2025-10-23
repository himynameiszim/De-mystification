import os
import re
import sys
import spacy
import pyinflect

def split_text_into_sentences(text: str) -> list[str]:
    """
    Splits a given text into sentences using regex.
    :param text: the input text to be split into sentences.
    :return: a list of sentences, where each sentence is a string.
    """
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
                text = text.replace('\n', '. ')
                sentences = split_text_into_sentences(text)
                # Use filename without extension as key, or keep full filename as key
                key = os.path.splitext(filename)[0]
                sentences_dict[key] = sentences
    return sentences_dict

def get_passive_subject(sentence: str) -> str:
    """
    Extract the grammatical subject of a passive sentence.
    :param sentence: a sentence in which the subject is in passive voice.
    :return: the subject of the passive sentence, or an empty string if not found.
    """
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(sentence)

    for token in doc:
        if token.dep_ == 'nsubjpass':
            return " ".join(i.text for i in token.subtree)
        
    return " "

def convert_passive_verb_to_active(passive_phrase: str) -> str:
    """
    Converts a passive verb phrase into its simple active form.
    :param passive_phrase: The string containing the passive verb phrase.
    :return: The converted active verb phrase as a string.
    """
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(passive_phrase)

    main_verb = None
    auxiliary_verb = None
    modal_verb = None

    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            main_verb = token
        elif "aux" in token.dep_ and token.pos_ == "AUX":
            if token.tag_ == "MD":
                modal_verb = token
            else:
                auxiliary_verb = token
    # no main verb found, but this is rare
    if not main_verb:
        return f"Could not identify a main verb in '{passive_phrase}'."

    verb_lemma = main_verb.lemma_

    # modal verb processing
    if modal_verb:
        return f"{modal_verb.text} {verb_lemma}"

    # auxiliary verb processing (this is optional)
    if auxiliary_verb:
        # check for past tense
        if auxiliary_verb.tag_ == "VBD":
            # past tense
            active_verb = nlp(verb_lemma)[0]._.inflect("VBD")
            return active_verb if active_verb else verb_lemma
        # base form
        else:
            return verb_lemma
        
    return verb_lemma

def extract_entity(text: str) -> list:
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(text)

    entities = list(set([ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'NORP']]))
    return entities if entities else ["NA"]

def get_agent_full_passive(text: str) -> str:
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(text)
    parts = text.rsplit(' by ', 1)

    if len(parts) > 1:
        agent_phrase = parts[1]
        doc = nlp(agent_phrase)
        # The first noun chunk is likely the main agent phrase.
        noun_chunks = list(doc.noun_chunks)
        if noun_chunks:
            core_agent = noun_chunks[0].text
            return core_agent.strip()
        else:
            # If spaCy finds no noun chunks, fall back to simple cleaning.
            cleaned_agent = re.sub(r'[.,?!;]+$', '', agent_phrase.strip())
            return cleaned_agent