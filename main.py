import os
import sys
import spacy
# import warnings

# warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# importing PassivePy
PassivePySRC_path = "path_toPassivePySrc"
if PassivePySRC_path not in sys.path:
    sys.path.append(PassivePySRC_path)

try:
    from PassivePySrc import PassivePy
except ImportError as e:
    print(f"Cannot import PassivePy: {e}")
    sys.exit(1)

from langchain_community.chat_models import ChatOpenAI

from modules import (
    read_txt_files_to_sentences_dict,
    split_text_into_sentences,
    PassiveDetectorAgent,
    ContextRetrieverAgent,
    AgentInferenceAgent,
    MystificationClassifierAgent,
    AgentClassifierAgent,
    AnnotatorAgent
)

def run_pipeline():
    # 2. init PassivePy
    try:
        passivepy = PassivePy.PassivePyAnalyzer(spacy_model = "en_core_web_lg")
        print(f"Loaded PassivePy model: {passivepy}\n")
    except Exception as e:
        print(f"Failed to load PassivePy. {e}\n")
        return
    
    # 3. init API key
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("No 'OPENAI_API_KEY' key found in environment variables.\n")
        return

    # 4. init LLM
    try:
        llm_gpt4o = ChatOpenAI(model_name="gpt-4o", temperature=0.1, openai_api_key=api_key)
        print(f"Loaded language model: {llm_gpt4o.name}\n")
    except Exception as e:
        print(f"Failed to load language model. {e}\n")
        return
    
    # 5. init agents
    try:
        passive_detector = PassiveDetectorAgent(passivepy_instance=passivepy)
        context_retriever = ContextRetrieverAgent(llm=llm_gpt4o, window_size=5)
        agent_inferencer = AgentInferenceAgent(llm=llm_gpt4o)
        mystification_classifier = MystificationClassifierAgent(llm=llm_gpt4o, text_input_window_size=5)
        agent_classifier = AgentClassifierAgent(llm=llm_gpt4o, passivepy_analyzer=passivepy, text_input_window_size=5)
        annotator = AnnotatorAgent()
        print("Loaded all agents.\n")
    except Exception as e:
        print(f"Failed to initialize agents. {e}\n")
        return
    
    # 6. input corpra
    input_dir = input("Enter corpra input directory: ").strip()
    if not os.path.isdir(input_dir):
        print(f"Invalid directory path: {input_dir}\n")
        return
    print(f"Reading files from: {input_dir}")
    sentences_dict = read_txt_files_to_sentences_dict(input_dir)
    if not sentences_dict:
        print("No text files found or files were empty in the specified directory.")
        return
    print(f"Loaded {len(sentences_dict)} file(s)\n")

    # 7. run pipeline
    print("...Doing my job...\n")
    print("...Running passive detector...")
    sentences_dict = passive_detector.run(sentences_dict)
    if not sentences_dict:
        print("No passive sentences detected. Exiting pipeline.\n")
        return
    print("...Running context retriever...")
    sentences_dict = context_retriever.run(sentences_dict)

    print("...Running agent inferencer...")
    sentences_dict = agent_inferencer.run(sentences_dict)

    print("...Running mystification classifier...")
    sentences_dict = mystification_classifier.run(sentences_dict)

    print("...Running agent classifier...")
    sentences_dict = agent_classifier.run(sentences_dict)

    print("...Running annotator...")
    output = annotator.run(sentences_dict)
    with open('output_wil.json', 'w', encoding='utf-8') as f:
        f.write(output)


    print("...Deed is done...!\n")
    print(f"Output is saved to 'output.json'.\n")
    f.close()

def main():
    run_pipeline()

if __name__ == "__main__":
    main()
