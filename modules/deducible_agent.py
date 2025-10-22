import json
from pprint import pprint
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.chains import LLMChain  

class DeducibleAgent:
    """
    Agent to assign a potential 'deduced agent' to passive sentences
    by using an LLM to match verbs against a predefined map.

    This agent processes a dictionary of sentences, identifies the passive ones,
    and uses an LLM to find the core verb in the verb phrase. It then matches
    this verb against the provided agent map to find the likely agent.
    """

    def __init__(self, llm: LLM): 
        super().__init__()     
        self.chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["sentence", "verb_phrase", "verb_list"],
                template=(
                    "You are a linguistic expert. Your task is to identify the main action verb "
                    "from a given verb phrase of a sentence and find its match from a provided list of verbs.\n\n"
                    "Sentence: \"{sentence}\"\n"
                    "Verb Phrase: \"{verb_phrase}\"\n\n"
                    "Verb List:\n---\n{verb_list}\n---\n\n"
                    "Based on the verb phrase, which verb from the list is similar?\n"
                    "If no verb in the list is a good match, respond with the word 'None'.\n"
                    "RESPOND ONLY WITH THE VERB, DO NOT ADD ADDITIONAL TEXT.\n\n"
                    "\n\nMatching Verb:"
                ),
            ),
        )

    def run(self, sentences_dict: dict, deducible_agent_map: dict) -> dict:
        """
        Processes the sentences_dict to find and assign deducible agents
        to passive sentences using LLM.

        :param sentences_dict: The dictionary of sentences to process.
        :return: The modified sentences_dict with 'deducible_agent' lists added.
        """
        verb_list_str = ", ".join(deducible_agent_map.keys())
        for filename, sentence_list in sentences_dict.items():
            for sentence_data in sentence_list:
                sentence_data['deducible_agent'] = []

                if sentence_data.get('voice_type') == "2": # we only want to process truncated passive, full passive return explicit agent anyway
                    verb_phrase_str = sentence_data.get('verb_phrase')
                    sentence_str = sentence_data.get('text', '' )
                    try:
                        # input_variables=["sentence", "verb_phrase", "verb_list"],
                        # Invoke the LLM chain to find the matching verb
                        matched_verb = self.chain.run(
                            sentence= sentence_str,
                            verb_phrase= verb_phrase_str,
                            verb_list= verb_list_str
                        )


                        # Check if the LLM's response is a valid key in our map
                        if matched_verb in deducible_agent_map:
                            deduced_agent = deducible_agent_map.get(matched_verb)
                            sentence_data['deducible_agent'].append(deduced_agent)
                        else:
                            # If no match found, append 'None' or an empty string
                            sentence_data['deducible_agent'].append('None')
                    
                    except Exception as e:
                        print(f"An error occurred during LLM invocation for verb phrase '{verb_phrase_str}': {e}")
        return sentences_dict