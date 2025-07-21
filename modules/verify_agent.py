import os
import sys

from langchain_core.language_models.llms import LLM 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class VerifierAgent:
    """
    An agent to verify whether or not the guessed agent of a passive sentence 
    is explicitly present or clearly co-referenced in the surrounding co-text.
    """
    def __init__(self, llm):
        """
        :param llm: An instance of a language model (e.g. ChatOpenAI for GPT-4o, Ollama, etc.).
        :param co_text_window_size: The number of sentences before and after the target sentence
                                    to consider as the 'co_text'. Defaults to 5.
        """
        self.llm = llm
        
        self.chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["guessed_agent", "co_text"],
                template=(
                    "You are a verification expert. Your task is to carefully read the text and determine if the given subject phrase can be infered (or appear) within it.\n\n"
                    "--- Input Data ---\n"
                    "Text:\n"
                    "\"\"\"\n{co_text}\n\"\"\"\n\n"
                    "--- Your Task ---\n"
                    "Does '{guessed_agent}' get stated or appear (might be inferred from) in the Text?\n"
                    "ANSWER ONLY with 'yes' or 'no'."
                ),
            ),
        )

    def run(self, sentences_dict: dict) -> dict:
        """
        Iterates through sentences_dict, and for each passive sentence with a valid guessed agent,
        verifies if that agent is present in the surrounding text. Adds the result as
        'agent_verification' to the sentence's dictionary.

        :param sentences_dict: A dictionary where keys are filenames and values are lists of 
                               sentence dictionaries. Expected keys include 'text', 'voice_type', and 'agent'.
        :return: The modified sentences_dict with 'agent_verification' added.
        """
        for filename, list_of_sentence_data_dicts in sentences_dict.items():
            for i, sentence_data in enumerate(list_of_sentence_data_dicts):
                if not isinstance(sentence_data, dict):
                    print(f"Warning: Expected a dictionary for sentence data in {filename} at index {i}. Skipping.")
                    continue
                
                # Verification is only applicable for passive sentences that have a guessed agent.
                voice_type = sentence_data.get('voice_type')
                
                # Check if this sentence is a candidate for verification
                if voice_type == '1':
                    sentence_data['agent_status'] = "explicit"
                    sentence_data['agent_verification'] = "yes"
                    sentence_data['mystification_idx'] = "1"  # Full Passive
                if sentence_data.get('guessed_agent') == "unknown":
                    sentence_data['agent_status'] = "unknown"
                    sentence_data['agent_verification'] = "NA"
                    sentence_data['mystification_idx'] = "3"  # Unknown agent
                if sentence_data.get('agent_status') == "other" and voice_type != '1':
                    sentence_data['agent_verification'] = "no"
                    sentence_data['mystification_idx'] = "3"  # Other agent
                elif voice_type == '2' and sentence_data.get('guessed_agent') != "unknown" and sentence_data.get('agent_status') != "other":
                    llm_input_co_text = sentence_data.get('co-text')
                    llm_input_guessed_agent = sentence_data.get('guessed_agent')

                    try:
                        result_str = self.chain.run(
                            guessed_agent=llm_input_guessed_agent,
                            co_text=llm_input_co_text
                        ).strip()

                        sentence_data['agent_verification'] = result_str
                    except Exception as e:
                        # print(f"Error during agent verification for sentence '{llm_input_target_sentence[:50]}...': {e}")
                        sentence_data['agent_verification'] = "error_in_processing"         
        return sentences_dict