import json

class AnnotatorAgent:
    """
    Agent to export the entire sentences_dict to a structured JSON string format.
    :param sentences_dict: Dictionary where keys are filenames and values are lists of sentence dictionaries, each sentence dictionary should contain JSON-serializable data 
                            (e.g., text, voice_type, context summary, agent_status, mystification_idx).
    :return: A JSON string representation of the sentences_dict.
    """
    def run(self, sentences_dict: dict) -> str:
        indent = 4
        if not isinstance(sentences_dict, dict):
            error_message = "Input to AnnotatorAgent.run must be a dictionary."
            print(f"Error: {error_message}")
            return json.dumps({"error": error_message, "type": "InputError"}, ensure_ascii=False, indent=indent)

        passive_sentences_to_export = {}

        for filename, list_of_sentence_data_dicts in sentences_dict.items():
            if not isinstance(list_of_sentence_data_dicts, list):
                print(f"Warning: Expected a list of sentences for file '{filename}', but got {type(list_of_sentence_data_dicts)}. This file will have an empty list in the output.")
                passive_sentences_to_export[filename] = []
                continue

            filtered_passive_sentences_for_file = []
            for sentence_data in list_of_sentence_data_dicts:
                # Ensure sentence_data is a dictionary and has 'voice_type'
                if not isinstance(sentence_data, dict):
                    print(f"Warning: Skipping non-dictionary item in '{filename}': {sentence_data}")
                    continue
                
                voice_type = sentence_data.get('voice_type')
                
                # Filter for passive sentences
                if voice_type in ['1', '2']:
                    filtered_passive_sentences_for_file.append(sentence_data)
            
            # Add the list of (only) passive sentences for this file to our output dictionary.
            # If a file has no passive sentences, it will be an empty list.
            passive_sentences_to_export[filename] = filtered_passive_sentences_for_file
        
        # Now, serialize the new dictionary which contains only passive sentences
        try:
            json_output_string = json.dumps(
                passive_sentences_to_export,
                ensure_ascii=False,  # Handles non-ASCII characters correctly
                indent=indent        # For pretty-printing
            )
            return json_output_string
        except TypeError as e:
            error_message = f"Data within the filtered passive sentences is not JSON serializable: {e}"
            print(f"Error: {error_message}")
            return json.dumps({"error": error_message, "type": "SerializationTypeError", "details": str(e)}, ensure_ascii=False, indent=indent)
        except Exception as e:
            error_message = f"An unexpected error occurred during JSON serialization: {e}"
            print(f"Error: {error_message}")
            return json.dumps({"error": error_message, "type": "UnexpectedSerializationError", "details": str(e)}, ensure_ascii=False, indent=indent)