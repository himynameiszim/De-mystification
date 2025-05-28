from langchain_core.language_models.llms import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class MystificationClassifierAgent:
    def __init__(self, llm, text_input_window_size: int = 5):
        """
        Initializes the MystificationClassifierAgent.

        :param llm: An initialized Langchain LLM instance (e.g., ChatOpenAI for GPT-4o).
        :param text_input_window_size: The number of original sentences before and after the
                                       current sentence to include in the 'text_window' input
                                       for the LLM prompt. Defaults to 5.
        """
        self.text_input_window_size = text_input_window_size
        self.chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["text_window", "voice_type", "context_summary", "agent_status"], # These are the variables the `run` method will provide
                template=(
                    "Your primary task is to assign a mystification index (level) to a specific TARGET SENTENCE. "
                    "This TARGET SENTENCE is embedded within the 'Text Window' provided below. "
                    "All other input information ('Voice Type', 'Summary of Surrounding Context', 'Determined Agent Status') refers directly to this TARGET SENTENCE.\n\n"
                    
                    "Mystification Index Definitions:\n"
                    "- '1': Stated and known (Agent is explicit).\n"
                    "- '2': Guessable with certainty (Agent is strongly implied by verb, world knowledge, or very strong immediate context).\n"
                    "- '3': Guessable with uncertainty (Agent is possibly recoverable from broader context, but not with certainty).\n"
                    "- '4': Mysterious and unknown (Specific agent/class of agent is not inferable from sentence or available context).\n\n"
                    
                    "Decision Guidance (apply to the TARGET SENTENCE):\n"
                    "1. If 'agent_status' is 'explicit' (this will be the case for voice_type '1' sentences), the Mystification Level MUST BE '1'. Your role then is to confirm this based on the provided data.\n"
                    "2. If 'agent_status' is 'implied' (typically for voice_type '2' sentences), the Mystification Level is usually '2'.\n"
                    "3. If 'agent_status' is 'unknown' (often for voice_type '2' sentences where context is unhelpful), the Mystification Level will be '3' or '4'. Distinguish based on whether the agent feels vaguely recoverable from the provided text window or context summary (level '3'), or if it feels completely obscure (level '4').\n\n"
                    
                    "Input Information for the TARGET SENTENCE:\n"
                    "Text Window (this window contains the TARGET SENTENCE you are analyzing): {text_window}\n"
                    "Voice Type of the TARGET SENTENCE (0: non-passive, 1: full passive, 2: truncated passive): {voice_type}\n"
                    "Summary of Surrounding Context (if available for the TARGET SENTENCE): {context_summary}\n"
                    "Determined Agent Status for the TARGET SENTENCE (explicit, implied, or unknown): {agent_status}\n\n"
                    
                    "Output ONLY the Mystification Level number (1, 2, 3, or 4) for the TARGET SENTENCE:"
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

            num_sentences_in_file = len(list_of_sentence_data_dicts)

            for i, current_sentence_data in enumerate(list_of_sentence_data_dicts):
                if not isinstance(current_sentence_data, dict):
                    print(f"Warning: Expected a dictionary for sentence data in {filename} at index {i}. Skipping this item.")
                    # Add a placeholder if the key must exist for all items
                    # current_sentence_data['mystification_idx'] = 'error_malformed_entry'
                    continue

                original_sentence_text = current_sentence_data.get('text', "[Original text missing]")
                voice_type_str = current_sentence_data.get('voice_type')
                agent_status_str = current_sentence_data.get('agent_status')
                
                display_sentence_text = original_sentence_text # Already defaults to a string above

                if voice_type_str == '0':  # Non-passive
                    current_sentence_data['mystification_idx'] = 'NA'
                if voice_type_str == '1':  # Full Passive
                    current_sentence_data['mystification_idx'] = '1'
                elif voice_type_str == '2':  # Truncated Passive - needs LLM processing
                    # 1. Construct the 'text_window' input for the prompt
                    text_window_start_idx = max(0, i - self.text_input_window_size)
                    text_window_end_idx = min(num_sentences_in_file, i + self.text_input_window_size + 1)
                    
                    text_window_parts = []
                    for k_idx in range(text_window_start_idx, text_window_end_idx):
                        sentence_dict_in_window = list_of_sentence_data_dicts[k_idx]
                        if isinstance(sentence_dict_in_window, dict) and 'text' in sentence_dict_in_window:
                            text_window_parts.append(sentence_dict_in_window.get('text', ''))
                        else:
                            text_window_parts.append("[Sentence data missing in window]")
                    
                    llm_input_text_window = " ".join(filter(None, text_window_parts)).strip()
                    if not llm_input_text_window: 
                        llm_input_text_window = original_sentence_text if original_sentence_text else "[No text available for window]"

                    llm_input_context_summary = current_sentence_data.get('context')
                    if llm_input_context_summary is None:
                        llm_input_context_summary = "No context summary was generated or available."
                    
                    current_llm_input_agent_status = agent_status_str if agent_status_str in ['explicit', 'implied', 'unknown'] else 'unknown'
                    if agent_status_str is None: 
                         current_llm_input_agent_status = 'unknown'

                    try:
                        result_str = self.chain.run(
                            text_window=llm_input_text_window,
                            voice_type=voice_type_str,
                            context_summary=llm_input_context_summary,
                            agent_status=current_llm_input_agent_status 
                        )
                        
                        mystification_idx = result_str.strip()
                        current_sentence_data['mystification_idx'] = mystification_idx
                            
                    except Exception as e:
                        print(f"Error during mystification classification for sentence '{display_sentence_text[:70]}...': {e}")
                        current_sentence_data['mystification_idx'] = "error_in_processing"
                        
        return sentences_dict