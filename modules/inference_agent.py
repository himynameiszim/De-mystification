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
                input_variables=["sentence", "verb_phrase", "cotext", "context", "entities_list", "guessed_agent", "deducible_list"],
                template=(
                    "You are analyzing a sentence for the presence of an agent (the doer of an action). "
                    "Based on the provided information determine if an agent is contextual, other, or unknown.\n"
                    "Input Details:\n"
                    "1. Target sentence: {sentence}\n"
                    "2. Verb phrase: {verb_phrase}\n"
                    "3. Co-text (surrounding sentences): {cotext}\n"
                    "4. Context (summary of the article containing the target sentence): {context}\n"
                    "5. Entity list: {entities_list}\n"
                    "6. Deduced agent list: {deducible_list}\n"
                    "7. Guessed agent of the verb phrase: {guessed_agent}\n"
                    "Guidance:\n"
                    "Your task is to determine whether the guessed agent of the verb phrase is: \n"
                    "- 'contextual' if the guessed agent is in the provided entity list or deduced agent list;\n"
                    "- 'other' if the guessed agent does not appear in the provided lists;\n"
                    "- 'unknown' if the guessed agent is unknown.\n"
                    "Note that the guessed agent can be paraphrased or not exactly match the entity in the list.\n\n"
                    "Answer ONLY with one of: 'contextual', 'other' or 'unknown'.\n"
                    "Agent Status ('contextual', 'other' or 'unknown'):"
                ),
            ),
        )

    def run(self, sentences_dict: dict) -> dict:
        for filename, list_of_sentence_data_dicts in sentences_dict.items():
            for sentence_data in list_of_sentence_data_dicts:

                sentence_text = sentence_data.get('text')
                voice_type_str = sentence_data.get('voice_type')
                verb_phrase_str = sentence_data.get('verb_phrase')
                text_window_str = sentence_data.get('co-text')
                context_str = sentence_data.get('context')
                entities_list_str = sentence_data.get('entities')
                guessed_agent_str = sentence_data.get('guessed_agent')
                deducible_list_str = sentence_data.get('deducible_agent')

                
                # Ensure sentence_text is available for logging or if needed by LLM
                # Default to empty string if None, to avoid errors with string operations like [:50]
                display_sentence_text = sentence_text if sentence_text is not None else ""

                if voice_type_str == '0':  # Non-Passive
                    sentence_data['agent_status'] = 'NA'
                    sentence_data['mystification_idx'] = '0'
                elif voice_type_str == '1':  # Full Passive
                    sentence_data['agent_status'] = 'explicit'
                    sentence_data['mystification_idx'] = '1'
                if guessed_agent_str == "unknown":
                    sentence_data['agent_status'] = 'unknown'
                    sentence_data['mystification_idx'] = 'NA'
                else:  # Truncated Passive
                    try:
                        # Call the LLM only for truncated passives
                        result_str = self.chain.run(
                            sentence=sentence_text,
                            verb_phrase=verb_phrase_str,
                            cotext=text_window_str,
                            context=context_str,
                            entities_list=entities_list_str,
                            guessed_agent=guessed_agent_str,
                            deducible_list=deducible_list_str
                        )
                        
                        agent_status = result_str.strip().lower()
                        sentence_data['agent_status'] = agent_status
                        if agent_status == 'contextual':
                            sentence_data['mystification_idx'] = '2'
                        elif agent_status == 'other' or agent_status == 'unknown':
                            sentence_data['mystification_idx'] = '3'
                        
                    except Exception as e:
                        print(f"Error during agent inference for truncated passive sentence '{display_sentence_text[:70]}...' in {filename}: {e}")
                        sentence_data['agent_status'] = "error_in_processing"
                        
        return sentences_dict