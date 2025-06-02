from langchain_core.language_models.llms import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class AgentClassifierAgent:
    """
    Agent to first extract passive verb phrases and then, for passive sentences,
    use an LLM to guess the agent (doer) of the action based on rich contextual information.
    """
    def __init__(self, llm, passivepy_analyzer, text_input_window_size: int = 5):
        """
        Initializes the AgentClassifierAgent.

        :param llm: An initialized Langchain LLM instance (e.g., ChatOpenAI for GPT-4o).
        :param passivepy_analyzer: An initialized instance of PassivePy.PassivePyAnalyzer.
        :param text_input_window_size: The number of original sentences before and after the
                                       current sentence to include in the 'text_window' input
                                       for the LLM prompt when guessing the agent. Defaults to 5.
        """

        self.llm = llm
        self.passivepy = passivepy_analyzer
        self.text_input_window_size = text_input_window_size

        prompt_str = (
            "You are an expert linguistic analyst. Your task is to identify the implied or stated agent (the doer of the action) "
            "for a given TARGET PASSIVE SENTENCE and its VERB PHRASE, using all available contextual information.\n\n"
            "Provided Information:\n"
            "1. TARGET PASSIVE SENTENCE: {target_sentence}\n"
            "2. Extracted PASSIVE VERB PHRASE from Target Sentence: {verb_phrase}\n"
            "3. Voice Type of Target Sentence (1: full passive, 2: truncated passive): {voice_type}\n"
            "4. Previously Determined Agent Status (explicit, implied, unknown): {agent_status}\n"
            "5. Previously Assigned Mystification Index for Target Sentence (1-4, or N/A): {mystification_idx}\n"
            "6. Summarized Context (summary of text originally surrounding the target sentence): {context_summary}\n"
            "7. Broader Text Window (original sentences surrounding and including the target sentence):\n"
            "   \"\"\"\n"
            "   {text_window}\n"
            "   \"\"\"\n\n"
            "Based on all the above information, who or what is the agent (the doer) performing the action of the verb phrase '{verb_phrase}' "
            "in the target sentence '{target_sentence}'?\n"
            "Provide a SHORT, ONLY the agent of the verb phrase. If the agent cannot be determined with reasonable certainty even with all the context, answer 'unknown'.\n"
            "Identified Agent:"
        )
        prompt = PromptTemplate(
            input_variables=[
                "target_sentence", "verb_phrase", "voice_type", "agent_status",
                "mystification_idx", "context_summary", "text_window"
            ],
            template=prompt_str
        )
        self.agent_guesser_chain = LLMChain(llm=self.llm, prompt=prompt)

    def run(self, sentences_dict: dict) -> dict:
        """
        Extracts passive verb phrases. For passive sentences, guesses the agent using an LLM.
        Adds 'verb_phrase' and 'agent' keys to the sentence dictionaries.

        :param sentences_dict: Dictionary where keys are filenames and values are lists of 
                               sentence dictionaries. Expected keys: 'text', 'voice_type'.
                               Uses 'context', 'agent_status', 'mystification_idx' if available.
        :return: The modified sentences_dict.
        """
        if not sentences_dict:
            return {}

        for filename, list_of_sentence_data_dicts in sentences_dict.items():
            if not isinstance(list_of_sentence_data_dicts, list):
                print(f"Warning: Expected a list of sentences for {filename}, but got {type(list_of_sentence_data_dicts)}. Skipping.")
                continue

            num_sentences_in_file = len(list_of_sentence_data_dicts)

            for i, sentence_data in enumerate(list_of_sentence_data_dicts):
                if not isinstance(sentence_data, dict):
                    print(f"Warning: Expected a dictionary for sentence data in {filename} at index {i}. Skipping this item.")
                    continue

                sentence_text = sentence_data.get('text')
                voice_type = sentence_data.get('voice_type')
                display_sentence_text = sentence_text if sentence_text is not None else "[No text]"


                # Part 1: Extract Passive Verb Phrase
                verb_phrase_str = "NA" # Default for non-passives or if text is missing
                if sentence_text: # Only proceed if there's text for PassivePy
                    try:
                        if voice_type == '1': # Full passive
                            verb_phrase_str = self.passivepy.match_text(sentence_text, full_passive=True, truncated_passive=False)["full_passive_matches"][0][0]
                        elif voice_type == '2': # Truncated passive
                            verb_phrase_str = self.passivepy.match_text(sentence_text, full_passive=False, truncated_passive=True)["truncated_passive_matches"][0][0]
                    except Exception as e:
                        print(f"Error extracting verb phrase for '{display_sentence_text[:70]}...': {e}")
                        verb_phrase_str = "Error extracting VP"
                sentence_data['verb_phrase'] = verb_phrase_str

                # Part 2: Guess the Agent (Doer) using LLM for Passive Sentences
                if voice_type == '0': # Non-Passive
                    sentence_data['agent'] = "NA"
                if voice_type in ['1', '2']:
                    if not sentence_text: # Should not happen if VP extraction happened
                        sentence_data['agent'] = "NA"
                        continue

                    # Prepare inputs for LLM
                    text_window_start_idx = max(0, i - self.text_input_window_size)
                    text_window_end_idx = min(num_sentences_in_file, i + self.text_input_window_size + 1)
                    
                    text_window_parts = []
                    for k_idx in range(text_window_start_idx, text_window_end_idx):
                        s_dict_in_window = list_of_sentence_data_dicts[k_idx]
                        if isinstance(s_dict_in_window, dict):
                            text_window_parts.append(s_dict_in_window.get('text', ''))
                        else:
                            text_window_parts.append("[Malformed sentence in window]")
                    
                    llm_input_text_window = " ".join(filter(None, text_window_parts)).strip()
                    if not llm_input_text_window: 
                        llm_input_text_window = sentence_text # Fallback

                    llm_input_target_sentence = sentence_text
                    llm_input_verb_phrase = verb_phrase_str # The one extracted above
                    llm_input_voice_type = voice_type
                    llm_input_agent_status = sentence_data.get('agent_status', 'unknown') # Use previously determined status
                    llm_input_mystification_idx = sentence_data.get('mystification_idx', 'NA') # Use previously determined index
                    llm_input_context_summary = sentence_data.get('context') # This is the summary
                    if llm_input_context_summary is None: 
                        llm_input_context_summary = "No specific context summary available."

                    try:
                        guessed_agent_str = self.agent_guesser_chain.run(
                            target_sentence=llm_input_target_sentence,
                            verb_phrase=llm_input_verb_phrase,
                            voice_type=llm_input_voice_type,
                            agent_status=llm_input_agent_status,
                            mystification_idx=llm_input_mystification_idx,
                            context_summary=llm_input_context_summary,
                            text_window=llm_input_text_window
                        ).strip()
                        
                        sentence_data['guessed_agent'] = guessed_agent_str if guessed_agent_str else "unknown"

                    except Exception as e:
                        print(f"Error during agent guessing for passive sentence '{display_sentence_text[:70]}...': {e}")
                        sentence_data['guessed_agent'] = "error_in_processing"
                        
        return sentences_dict