from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class AgentInferenceAgent:
    """
    Agent to evaluate whther an agent (do-er) is present or implied in a given passive sentence with its context.
    :param llm: An instance of a language model (LLM) to use for inference.
    :param sentences_dict: A dictionary where keys are filenames and values are lists of 'sentences', 'voice_type', 'context' and appended 'agent_status'.
    :return 
    """
    def __init__(self, llm):
        
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
        )
        prompt = ChatPromptTemplate.from_template(template)
        self.chain = prompt | llm | StrOutputParser()

    def run(self, sentences_dict: dict) -> dict:
        for filename, list_of_sentence_data_dicts in sentences_dict.items():

            # batching attempt here
            batch_inputs = []
            sentences_to_update = []
            for _, sentence_data in enumerate(list_of_sentence_data_dicts):
                if not isinstance(sentence_data, dict):
                    print(f"Expected a dictionary for sentence data in {filename}. Skipping this item.")
                    continue
                voice_type_str = sentence_data.get('voice_type')
                if voice_type_str == '0':  # Non-Passive
                    sentence_data['agent_status'] = 'NA'
                elif voice_type_str == '1':  # Full Passive
                    sentence_data['agent_status'] = 'explicit'
                else:  # Truncated Passive
                    llm_inputs = {
                        "sentence": sentence_data.get('text'),
                        "verb_phrase": sentence_data.get('verb_phrase'),
                        "cotext": sentence_data.get('co-text'),
                        "context": sentence_data.get('context'),
                        "entities_list": sentence_data.get('entities'),
                        "guessed_agent": sentence_data.get('guessed_agent'),
                        "deducible_list": sentence_data.get('deducible_agent')
                    }
                    batch_inputs.append(llm_inputs)
                    sentences_to_update.append(sentence_data)
            if batch_inputs:
                try:
                    statusses = self.chain.batch(
                        batch_inputs,
                        config={"return_exceptions": True}
                    )
                except Exception as e:
                    print(f"Error during batching agent inference for file '{filename}': {e}")
                    statusses = [e] * len(batch_inputs)

                for sentence_data, status in zip(sentences_to_update, statusses):
                    if isinstance(status, Exception):
                        display_text = sentence_data.get('text', '[No text]')[:70]
                        print(f"Error during batched agent inference for sentence '{display_text}...': {status}")
                        sentence_data['agent_status'] = "NA"
                    else:
                        agent_status = status.strip().lower()
                        sentence_data['agent_status'] = agent_status
        return sentences_dict