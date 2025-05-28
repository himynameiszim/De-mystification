class PassiveDetectorAgent:
    """
    Agent to detect full and truncated passive sentences in a given set of sentences (input as a dictionary).
    :param self: Instance of the PassivePyAnalyzer class (read PassivePy.py).
    :param sentences_dict: Dictionary where keys are filenames and values are lists of sentences of the corresponding files.
    :return: sentences_dict: the same dictionary as input but the second index of values corresponding to each key is assigned a value:
                            '0': non-passive sentences
                            '1': full-passive sentences
                            '2': truncated-passive sentences
    """
    def __init__(self, passivepy_instance):
        self.passivepy = passivepy_instance

    def run(self, sentences_dict):
        for filename, sentences_data in sentences_dict.items():
            processed_sentences_for_file = []
            sentences_list_to_process = []

            if isinstance(sentences_data, str):
                sentences_list_to_process = [sentences_data]
            elif isinstance(sentences_data, (list, tuple)):
                sentences_list_to_process = sentences_data

            for sentence_item in sentences_list_to_process:
                sentence_text = ""
                voice_type = '0' #default is non-passive

                sentence_text = sentence_item
                
                doc = self.passivepy.nlp(sentence_text)

                # check for full passive
                full_match = self.passivepy._find_unique_spans(doc, truncated_passive=False, full_passive=True)
                if full_match:
                    voice_type = '1'
                else:
                    truncated_match = self.passivepy._find_unique_spans(doc, truncated_passive=True, full_passive=False)
                    if truncated_match:
                       voice_type = '2'

                processed_sentences_for_file.append([sentence_text, voice_type])
            
            sentences_dict[filename] = processed_sentences_for_file

        return sentences_dict