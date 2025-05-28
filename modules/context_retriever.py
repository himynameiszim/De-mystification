from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

class ContextRetrieverAgent:
    """
    Agent to retrieve context surrounding a sentence as marked as "passive" (either full or truncated).
    Default surrounding text (window_size) is set to be 5 sentences. Default llm is OpenAI's GPT-4o.
    :param: llm: An instance of a language model (LLM) to use for summarization.
    :param: window_size: Number of sentences to include before and after the current sentence for context.
    :return: sentences_dict: the same dictionary as input but append the 'context' value to each 'text' value (if it is passive).
    """
    def __init__(self, llm: LLM, window_size: int = 5):
        self.llm = llm
        self.window_size = window_size

        prompt_template_str = (
            "You are an expert at summarizing text. Below is a collection of sentences providing context "
            "around a central sentence. Please provide a detailed summary of this entire context. "
            "The central sentence itself is included in this text.\n\n"
            "Context Text:\n\"\"\"\n{context_text}\n\"\"\"\n\n"
            "Detailed Summary:"
        )
        prompt_template = PromptTemplate.from_template(prompt_template_str)
        
        self.summarization_chain = RunnableSequence(
            prompt_template | self.llm
        )

    def run(self, sentences_dict: dict) -> dict:
        for filename, sentence_list_from_passive_detector in sentences_dict.items():
            num_sentences_in_file = len(sentence_list_from_passive_detector)
            # This new list will hold dictionaries instead of lists
            processed_file_entries = [] 

            for i, sentence_entry in enumerate(sentence_list_from_passive_detector):

                current_sentence_text = sentence_entry[0]
                voice_type = sentence_entry[1]
                
                # Initialize the dictionary for the current sentence.
                # This will be the new structure for all sentences in the output.
                output_sentence_data = {
                    'text': current_sentence_text,
                    'voice_type': voice_type,
                    'context': None  # Default context is None
                }
                
                if voice_type in ['1', '2']:  # Process only passive sentences for summarization
                    start_index = max(0, i - self.window_size)
                    end_index_for_slicing_after = min(num_sentences_in_file, i + self.window_size + 1) 

                    # Retrieve surrounding texts
                    sentences_before_texts = [
                        s[0] for s in sentence_list_from_passive_detector[start_index:i]
                        if isinstance(s, list) and len(s) > 0 # Ensure 's' is a list and has text
                    ]
                    sentences_after_texts = [
                        s[0] for s in sentence_list_from_passive_detector[i+1:end_index_for_slicing_after]
                        if isinstance(s, list) and len(s) > 0 # Ensure 's' is a list and has text
                    ]
                    
                    context_texts_to_summarize = sentences_before_texts + [current_sentence_text] + sentences_after_texts
                    full_context_string = " ".join(filter(None, context_texts_to_summarize)).strip()

                    if full_context_string:
                        try:
                            # Invoke the summarization chain
                            summary_result = self.summarization_chain.invoke({'context_text': full_context_string})
                            
                            summary_text = ""
                            if hasattr(summary_result, 'content'):  # For AIMessage from ChatModels
                                summary_text = summary_result.content
                            elif isinstance(summary_result, str): # For simpler LLM outputs
                                summary_text = summary_result
                                
                            output_sentence_data['context'] = summary_text.strip()
                            
                        except Exception as e:
                            print(f"Error during summarization for sentence '{current_sentence_text[:50]}...' in {filename}: {e}")
                            output_sentence_data['context'] = "Error during summarization."
                    else:
                        output_sentence_data['context'] = "Context string was empty; no summary generated."
                
                processed_file_entries.append(output_sentence_data)
            
            # Replace the list of lists with the new list of dictionaries for the current file
            sentences_dict[filename] = processed_file_entries
            
        return sentences_dict