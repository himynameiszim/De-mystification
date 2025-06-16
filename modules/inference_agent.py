from langchain_core.language_models.llms import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain    

class AgentInferenceAgent:
    """
    Agent to evaluate whther an agent (do-er) is present or implied in a given passive sentence with its context.
    :param llm: An instance of a language model (LLM) to use for inference.
    :param sentences_dict: A dictionary where keys are filenames and values are lists of 'sentences', 'voice_type', 'context' and appended 'agent_status'.
    :return 
    """
    def __init__(self, llm):
        super().__init__()
        self.chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["sentence", "verb_phrase", "cotext", "context", "entities_list", "guessed_agent"],
                template=(
                    "You are analyzing a sentence for the presence of an agent (the doer of an action). "
                    "Based on the provided sentence, its surrounding context, the guessed agent and the list of possible agents infered from the context, "
                    "determine if an agent is implied, or unknown.\n"
                    "Guidance based on voice type:\n"
                    "Your task is to determine whether the guessed agent of the verb phrase is: - 'implied' if the guessed agent is in the provided entity list; - 'unknown' if the guessed agent is not in the provided entity list or 'unknown'.\n"
                    "Note that the guessed agent can be paraphrased or not exactly match the entity in the list, so use your common sense.\n\n"
                    "Input Details:\n"
                    "1. Target sentence: {sentence}\n"
                    "2. Verb phrase: {verb_phrase}\n"
                    "3. Co-text (surrounding sentences): {cotext}\n"
                    "4. Context (summary of the article containing the target sentence): {context}\n"
                    "5. Entity list: {entities_list}\n"
                    "6. Guessed agent of the verb phrase: {guessed_agent}\n"
                    "Answer ONLY with one of: 'implied', 'unknown'.\n"
                    "Agent Status (implied, or unknown):"
                ),
            ),
        )

    def run(self, sentences_dict: dict) -> dict:
        for filename, list_of_sentence_data_dicts in sentences_dict.items():
            for sentence_data in list_of_sentence_data_dicts:

                sentence_text = sentence_data.get('text')
                voice_type_str = sentence_data.get('voice_type')
                verb_phrase_str = sentence_data.get('verb_phrase')
                
                # Ensure sentence_text is available for logging or if needed by LLM
                # Default to empty string if None, to avoid errors with string operations like [:50]
                display_sentence_text = sentence_text if sentence_text is not None else ""

                if voice_type_str == '0':  # Non-Passive
                    sentence_data['agent_status'] = 'NA'
                elif voice_type_str == '1':  # Full Passive
                    sentence_data['agent_status'] = 'explicit'
                elif voice_type_str == '2':  # Truncated Passive - needs LLM processing
                    # input_variables=["sentence", "verb_phrase", "cotext", "context", "entities_list", "guessed_agent"],
                    try:
                        # Call the LLM only for truncated passives
                        result_str = self.chain.run(
                            sentence=sentence_text,
                            verb_phrase=verb_phrase_str,
                            cotext=sentence_data.get('co-text'),
                            context=sentence_data.get('context'),
                            entities_list=sentence_data.get('entities'),
                            guessed_agent=sentence_data.get('guessed_agent')
                        )
                        
                        agent_status = result_str.strip().lower()
                        sentence_data['agent_status'] = agent_status
                        if agent_status == 'implied':
                            sentence_data['mystification_idx'] = '2'
                        elif agent_status == 'unknown':
                            sentence_data['mystification_idx'] = '3'
                        
                    except Exception as e:
                        print(f"Error during agent inference for truncated passive sentence '{display_sentence_text[:70]}...' in {filename}: {e}")
                        sentence_data['agent_status'] = "error_in_processing"
                        
        return sentences_dict