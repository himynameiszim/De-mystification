from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .utils import get_passive_subject, convert_passive_verb_to_active, get_agent_full_passive

class AgentClassifierAgent:
    """
    Agent to first extract passive verb phrases and then, for passive sentences,
    use an LLM to guess the agent (doer) of the action based on contextual information.
    """
    def __init__(self, llm, passivepy_analyzer):
        """
        Initializes the AgentClassifierAgent.

        :param llm: An initialized Langchain LLM instance (e.g., ChatOpenAI for GPT-4o).
        :param passivepy_analyzer: An initialized instance of PassivePy.PassivePyAnalyzer.
        """

        self.llm = llm
        self.passivepy = passivepy_analyzer

        prompt_str = (
            "You are an expert linguistic analyst. Your task is to identify doer of an action in a passive voice sentence.\n"
            "1. TARGET PASSIVE SENTENCE: {target_sentence}\n"
            "2. Extracted PASSIVE VERB PHRASE from Target Sentence: {verb_phrase}\n"
            "3. Context: {context_summary}\n"
            "4. Broader Text Window (original sentences is the last sentence):\n"
            "   \"\"\"\n"
            "   {text_window}\n"
            "   \"\"\"\n\n"
            "5. List of entities in the co-text: {entities_list}\n" \
            "6. Deducible agent list: {deducible_list}\n"
            "Based on all the above information, who or what perform the action described by the verb phrase '{verb_phrase}'\n"
            "in the target sentence '{target_sentence}'?\n"
            "If the agent is present in the provided deducible agent list, LINK IT TO THE ENTITIES APPEARED IN THE PROVIDED ENTITIES LIST.\n"
            "If none of the entities in the provided lists cannot possibly perform the given action, use common knowledge.\n"
            "If the agent cannot be determined with reasonable certainty even with all the context and common knowledge, answer 'unknown'.\n\n"
            "ANSWER WITH ONLY ONE AGENT OR UNKNOWN. DO NOT ADD ADDITIONAL TEXT OR REASONING.\n"
            "Guessed Agent:"
        )
        prompt = ChatPromptTemplate.from_template(prompt_str)
        self.agent_guesser_chain = prompt | self.llm | StrOutputParser()

    def run(self, sentences_dict: dict) -> dict:
        """
        For passive sentences, guesses the agent using an LLM.
        Adds 'verb_phrase' and 'agent' keys to the sentence dictionaries.

        :param sentences_dict: Dictionary where keys are filenames and values are lists of 
                               sentence dictionaries. Expected keys: 'text', 'voice_type'.
                               Uses 'context', 'agent_status', 'mystification_idx' if available.
        :return: The modified sentences_dict.
        """
        if not sentences_dict:
            return {}

        for filename, list_of_sentence_data_dicts in sentences_dict.items():
            # batching attempt here
            batch_inputs = []
            sentences_to_update = []

            for i, sentence_data in enumerate(list_of_sentence_data_dicts):
                if not isinstance(sentence_data, dict):
                    print(f"Warning: Expected a dictionary for sentence data in {filename} at index {i}. Skipping this item.")
                    continue
                voice_type = sentence_data.get('voice_type')
                if voice_type == '0': # Non-Passive
                    sentence_data['guessed_agent'] = "NA"
                if voice_type == '1': # Full-Passive
                    sentence_data['guessed_agent'] = get_agent_full_passive(sentence_data['text'])
                elif voice_type == '2':
                    llm_inputs = {
                        "target_sentence": sentence_data.get('text'),
                        "verb_phrase": sentence_data.get('verb_phrase'),
                        "context_summary": sentence_data.get('context'),
                        "text_window": sentence_data.get('co-text'),
                        "entities_list": sentence_data.get('entities'),
                        "deducible_list": sentence_data.get('deducible_agent')
                    }
                    batch_inputs.append(llm_inputs)
                    sentences_to_update.append(sentence_data)
            if batch_inputs:
                try:
                    guessed_agents = self.agent_guesser_chain.batch(batch_inputs, config={"return_exceptions": True})

                    for sentence_data, guessed_agent in zip(sentences_to_update, guessed_agents):
                        if isinstance(guessed_agent, Exception):
                            display_text = sentence_data.get('text', '[No text]')[:70]
                            print(f"Error during batched agent guessing for sentence '{display_text}...': {guessed_agent}")
                            sentence_data['guessed_agent'] = "error_in_processing"
                        else:
                            guessed_agent = guessed_agent.strip()
                            sentence_data['guessed_agent'] = guessed_agent if guessed_agent else "unknown"
                except Exception as e:
                    print(f"Error during batched agent guessing in file '{filename}: {e}")
                    for sentence_data in sentences_to_update:
                        sentence_data['guessed_agent'] = "NA"
        return sentences_dict