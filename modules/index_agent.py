from langchain_core.language_models.llms import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class MystificationClassifierAgent:
    def __init__(self, llm):
        """
        Initializes the MystificationClassifierAgent.

        :param llm: An initialized Langchain LLM instance (e.g., ChatOpenAI for GPT-4o).
        """
        self.chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["text", "text_window", "verb_phrase", "context_summary", "agent_status", "guessed_agent"], # These are the variables the `run` method will provide
                template=(
                    "Your primary task is to assign a mystification level to a specific TARGET SENTENCE. "
                    "This TARGET SENTENCE is the last sentence within the 'Text Window' provided below.\n\n"
                    
                    "Mystification Index Definitions:\n"
                    "- '1': Stated and known (Agent is explicit).\n"
                    "- '2': Guessable with certainty (Agent is strongly implied by verb, world knowledge, or very strong immediate context).\n"
                    "- '3': Mysterious and unknown (Agent is not recoverable from broader context or common knowledge).\n"
                    
                    "Decision Guidance (apply to the TARGET SENTENCE):\n"
                    "2. If 'agent_status' is 'contextual': the Mystification Level will be '2'.\n"
                    "3. If 'agent_status' is 'unknown': the Mystification Level will be '3'.\n\n"
                    
                    "Input Information:\n"
                    "Target sentence (the sentence you are analyzing): {text}\n"
                    "Text Window (this window contains the TARGET SENTENCE you are analyzing): {text_window}\n"
                    "Extracted Verb Phrase: {verb_phrase}\n"
                    "Summary of Surrounding Context: {context_summary}\n"
                    "Guessed Agent of the Verb Phrase: {guessed_agent}\n"
                    "Determined Agent Status for the TARGET SENTENCE (explicit, implied, or unknown): {agent_status}\n\n"
                    
                    "Output ONLY the Mystification Level number (2, or 3) for the TARGET SENTENCE:"
                ),
            ),
        )

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

            for i, current_sentence_data in enumerate(list_of_sentence_data_dicts):
                if not isinstance(current_sentence_data, dict):
                    print(f"Warning: Expected a dictionary for sentence data in {filename} at index {i}. Skipping this item.")
                    # Add a placeholder if the key must exist for all items
                    # current_sentence_data['mystification_idx'] = 'error_malformed_entry'
                    continue

                original_sentence_text = current_sentence_data.get('text')
                voice_type_str = current_sentence_data.get('voice_type')
                verb_phrase_str = current_sentence_data.get('verb_phrase')
                agent_status_str = current_sentence_data.get('agent_status')
                guessed_agent_str = current_sentence_data.get('guessed_agent')
                
                display_sentence_text = original_sentence_text # Already defaults to a string above

                if voice_type_str == '0':  # Non-passive
                    current_sentence_data['mystification_idx'] = 'NA'
                if voice_type_str == '1':  # Full Passive
                    current_sentence_data['mystification_idx'] = '1'
                elif voice_type_str == '2':  # Truncated Passive - needs LLM processing
                    llm_input_text_window = current_sentence_data.get('co-text')
                    llm_input_context_summary = current_sentence_data.get('context')
                    if llm_input_context_summary is None:
                        llm_input_context_summary = "No context summary was generated or available."
                    if llm_input_text_window is None:
                        llm_input_text_window = "No text window was provided or available."
                    
                    current_llm_input_agent_status = agent_status_str if agent_status_str in ['explicit', 'contextual', 'unknown'] else 'unknown'
                    if agent_status_str is None: 
                        current_llm_input_agent_status = 'unknown'

                    try:
                        result_str = self.chain.run(
                            text=original_sentence_text,
                            text_window=llm_input_text_window,
                            voice_type=voice_type_str,
                            verb_phrase=verb_phrase_str,
                            context_summary=llm_input_context_summary,
                            agent_status=current_llm_input_agent_status,
                            guessed_agent=guessed_agent_str
                        )
                        
                        mystification_idx = result_str.strip()
                        current_sentence_data['mystification_idx'] = mystification_idx
                            
                    except Exception as e:
                        print(f"Error during mystification classification for sentence '{display_sentence_text[:70]}...': {e}")
                        current_sentence_data['mystification_idx'] = "error_in_processing"
                        
        return sentences_dict