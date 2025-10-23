from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class MystificationClassifierAgent:
    def __init__(self, llm):
        """
        Initializes the MystificationClassifierAgent.

        :param llm: An initialized Langchain LLM instance (e.g., ChatOpenAI for GPT-4o).
        """
        template=(
            "Your primary task is to assign a mystification level to a specific TARGET SENTENCE.\n"            
            "Mystification Index Definitions:\n"
            "- '2': Guessable with certainty (Agent is strongly implied by verb, world knowledge, or very strong immediate context).\n"
            "- '3': Mysterious and unknown (Agent is not recoverable from broader context or common knowledge).\n"
            "Input Information:\n"
            "Target sentence (the sentence you are analyzing): {text}\n"
            "Text Window (this window contains the TARGET SENTENCE you are analyzing): {text_window}\n"
            "Extracted Verb Phrase: {verb_phrase}\n"
            "Summary of Surrounding Context: {context_summary}\n"
            "Guessed Agent of the Verb Phrase: {guessed_agent}\n"
            "Determined Agent Status for the TARGET SENTENCE (implied, or unknown): {agent_status}\n\n"
            
            "OUTPUT ONLY THE MYSTIFICATION NUMBER (2 OR 3)FOR THE TARGET SENTENCE. DO NOT ADD ADDITIONAL REASONING OR TEXT."
        )
        prompt = ChatPromptTemplate.from_template(template)
        self.chain = prompt | llm | StrOutputParser()

    def run(self, sentences_dict: dict) -> dict:
        """
        Iterates through sentences_dict.
        - If voice_type is '1', mystification_idx is '1'.
        - If voice_type is '2', uses an LLM to determine the mystification_idx.
        - If voice_type is '0' (or other), mystification_idx is 'N/A (Non-Passive/Undefined Voice Type)'.
        Appends 'mystification_idx' to each sentence's dictionary.

        :param sentences_dict: Dictionary where keys are filenames and values are lists of 
                               sentence dictionaries. Each sentence dictionary is expected to have 
                               'text' (original sentence), 'voice_type', 'context' (summary), 
                               and 'agent_status'.
        :return: The modified sentences_dict with 'mystification_idx' added.
        """
        for filename, list_of_sentence_data_dicts in sentences_dict.items():
            if not isinstance(list_of_sentence_data_dicts, list):
                print(f"Warning: Expected a list of sentences for {filename}, but got {type(list_of_sentence_data_dicts)}. Skipping.")
                continue

            batch_inputs = []
            sentences_to_update = []

            for i, current_sentence_data in enumerate(list_of_sentence_data_dicts):
                if not isinstance(current_sentence_data, dict):
                    print(f"Warning: Expected a dictionary for sentence data in {filename} at index {i}. Skipping this item.")
                    continue
                voice_type_str = current_sentence_data.get('voice_type')
                if voice_type_str == '0':  # Non-passive
                    current_sentence_data['mystification_idx'] = 'NA'
                if voice_type_str == '1':  # Full Passive
                    current_sentence_data['mystification_idx'] = '1'
                elif voice_type_str == '2':  # Truncated Passive - needs LLM processing
                    llm_inputs = {
                    "text": current_sentence_data.get('text'),
                    "text_window": current_sentence_data.get('co-text'),
                    "voice_type": voice_type_str,
                    "verb_phrase": current_sentence_data.get('verb_phrase'),
                    "context_summary": current_sentence_data.get('context'),
                    "agent_status": current_sentence_data.get('agent_status'),
                    "guessed_agent": current_sentence_data.get('guessed_agent')
                    }

                    batch_inputs.append(llm_inputs)
                    sentences_to_update.append(current_sentence_data)
                else:
                    current_sentence_data['mystification_idx'] = "NA"

            if batch_inputs:
                try:
                    mystification_idxs = self.chain.batch(
                        batch_inputs,
                        config={"return_exceptions": True}
                    )
                except Exception as e:
                    print(f"Error during batched mystification index assignment in file '{filename}': {e}")
                    mystification_idxs = [e] * len(batch_inputs)
                
                for sentence_data, mystification_idx in zip(sentences_to_update, mystification_idxs):
                    if isinstance(mystification_idx, Exception):
                        display_text = sentence_data.get('text', '[No text]')[:70]
                        print(f"Error during mystification index assignment for sentence '{display_text}...': {mystification_idx}")
                        sentence_data['mystification_idx'] = "NA"
                    else:
                        mystification_idx = mystification_idx.strip()
                        sentence_data['mystification_idx'] = mystification_idx        
        return sentences_dict