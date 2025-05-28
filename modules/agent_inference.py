from langchain_core.language_models.llms import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain    

class AgentInferenceAgent:
    """
    Agent to evaluate whther an agent (do-er) is present or implied in a given sentence with its context.
    :param llm: An instance of a language model (LLM) to use for inference.
    :param sentences_dict: A dictionary where keys are filenames and values are lists of 'sentences', 'voice_type', 'context' and appended 'agent_status'.
    :return 
    """
    def __init__(self, llm):
        super().__init__()
        self.chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["sentence", "voice_type", "context"],
                template=(
                    "You are analyzing a sentence for the presence of an agent (the doer of an action). "
                    "Based on the provided sentence, its voice type, and its surrounding context, "
                    "determine if an agent is explicitly mentioned, implied, or unknown."
                    "Some sentences will require common world knowledge. Also, you have to understand the context to determine the agent status.\n\n"
                    "Answer ONLY with one of: 'explicit', 'implied', 'unknown'.\n\n"
                    "Guidance based on voice type:\n"
                    "- If voice type is '1' (full passive), the agent is typically mentioned (e.g., 'by someone'). Immediately answer with 'explicit'.\n"
                    "- If voice type is '2' (truncated passive), the agent is not mentioned in the sentence itself. Your task is to determine if it's implied by the context or truly unknown ('implied' or 'unknown').\n\n"
                    "Input Details:\n"
                    "Voice type (0: non-passive, 1: full passive, 2: truncated passive): {voice_type}\n"
                    "Sentence: {sentence}\n"
                    "Context: {context}\n\n"
                    "Agent Status (explicit, implied, or unknown):"
                ),
            ),
        )

    def run(self, sentences_dict: dict) -> dict:
        for filename, list_of_sentence_data_dicts in sentences_dict.items():
            for sentence_data in list_of_sentence_data_dicts:

                sentence_text = sentence_data.get('text')
                voice_type_str = sentence_data.get('voice_type')
                
                # Ensure sentence_text is available for logging or if needed by LLM
                # Default to empty string if None, to avoid errors with string operations like [:50]
                display_sentence_text = sentence_text if sentence_text is not None else ""

                if voice_type_str == '0':  # Non-Passive
                    sentence_data['agent_status'] = 'NA'
                elif voice_type_str == '1':  # Full Passive
                    sentence_data['agent_status'] = 'explicit'
                elif voice_type_str == '2':  # Truncated Passive - needs LLM processing
                    if sentence_text is None: # Sentence text is crucial for the LLM
                        print(f"Warning: Missing 'text' for a voice_type '2' sentence in {filename}. Assigning 'unknown' agent_status.")
                        sentence_data['agent_status'] = 'unknown'
                        continue

                    context_str = sentence_data.get('context')
                    current_context = context_str if context_str is not None else "No additional context provided."
                    
                    try:
                        # Call the LLM only for truncated passives
                        result_str = self.chain.run(
                            sentence=sentence_text,
                            voice_type=voice_type_str,  # Will be '2'
                            context=current_context
                        )
                        
                        agent_status = result_str.strip().lower()
                        sentence_data['agent_status'] = agent_status
                        
                    except Exception as e:
                        print(f"Error during agent inference for truncated passive sentence '{display_sentence_text[:70]}...' in {filename}: {e}")
                        sentence_data['agent_status'] = "error_in_processing"
                        
        return sentences_dict