from langchain_core.language_models.llms import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from .utils import get_passive_subject, convert_passive_verb_to_active

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
            "You are an expert linguistic analyst. Your task is to identify doer of an action"
            "for a given TARGET PASSIVE SENTENCE and its VERB PHRASE, using all available contextual information.\n"
            "You will be privided with the list of all entities in the context of the target sentence.\n"
            "Note that some sentences require common knowledge other than choosing an entity from the provided list.\n\n"
            "Provided Information:\n"
            "1. TARGET PASSIVE SENTENCE: {target_sentence}\n"
            "2. Voice Type of the Target Sentence: {voice_type} (1 - full passive, 2 - truncated passive)\n"
            "3. Extracted PASSIVE VERB PHRASE from Target Sentence: {verb_phrase}\n"
            "4. Summarized Context (summary of the article containing the target passive sentence): {context_summary}\n"
            "5. Broader Text Window (original sentences is the last sentence):\n"
            "   \"\"\"\n"
            "   {text_window}\n"
            "   \"\"\"\n\n"
            "6. List of entities in the co-text: {entities_list}\n"
            "Based on all the above information, who or what '{active_verb_phrase}' the subject '{subject_of_passive}' "
            "in the target sentence '{target_sentence}'?\n"
            "If the voice type is '1', answer with the agent of the verb phrase directly after 'by'.\n"
            "If the voice type is '2', you should use the verb phrase and the context to make a guess.\n"
            "If none of the entities in the provided list cannot possibly perform the given action, use your common sense (e.g: the house was broken into -> answer with 'criminal').\n"
            "If the agent cannot be determined with reasonable certainty even with all the context and common knowledge, answer 'unknown'.\n"
            "Provide ONLY one agent of the verb phrase.\n"
            "Guessed Agent:"
        )
        prompt = PromptTemplate(
            input_variables=[
                "target_sentence", "voice_type", "verb_phrase",  "context_summary", "text_window", "entities_list", "active_verb_phrase", "subject_of_passive"
            ],
            template=prompt_str
        )
        self.agent_guesser_chain = LLMChain(llm=self.llm, prompt=prompt)

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
            for i, sentence_data in enumerate(list_of_sentence_data_dicts):
                if not isinstance(sentence_data, dict):
                    print(f"Warning: Expected a dictionary for sentence data in {filename} at index {i}. Skipping this item.")
                    continue

                sentence_text = sentence_data.get('text')
                voice_type = sentence_data.get('voice_type')
                verb_phrase_str = sentence_data.get('verb_phrase')
                display_sentence_text = sentence_text if sentence_text is not None else "[No text]"

                # Part 2: Guess the Agent (Doer) using LLM for Passive Sentences
                if voice_type == '0': # Non-Passive
                    sentence_data['guessed_agent'] = "NA"
                if voice_type in ['1', '2']:

                    llm_input_text_window = sentence_data.get('co-text')
                    llm_input_target_sentence = sentence_text
                    llm_input_subject_of_passive = get_passive_subject(sentence_text)
                    llm_input_verb_phrase = verb_phrase_str
                    llm_input_active_verb_phrase = convert_passive_verb_to_active(verb_phrase_str)
                    llm_input_voice_type = voice_type
                    llm_input_context_summary = sentence_data.get('context')
                    llm_input_entities_list = sentence_data.get('entities')


                    try:
                        guessed_agent_str = self.agent_guesser_chain.run(
                            target_sentence=llm_input_target_sentence,
                            voice_type=llm_input_voice_type,
                            verb_phrase=llm_input_verb_phrase,
                            context_summary=llm_input_context_summary,
                            text_window=llm_input_text_window,
                            entities_list=llm_input_entities_list,
                            active_verb_phrase=llm_input_active_verb_phrase,
                            subject_of_passive=llm_input_subject_of_passive
                        ).strip()
                        
                        sentence_data['guessed_agent'] = guessed_agent_str if guessed_agent_str else "unknown"

                    except Exception as e:
                        print(f"Error during agent guessing for passive sentence '{display_sentence_text[:70]}...': {e}")
                        sentence_data['guessed_agent'] = "error_in_processing"
                        
        return sentences_dict