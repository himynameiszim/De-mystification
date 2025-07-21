import os
import sys
import spacy
import pyinflect
import json
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
from langchain_community.chat_models import ChatOllama

from modules import (
    read_txt_files_to_sentences_dict,
    split_text_into_sentences,
    get_passive_subject,
    convert_passive_verb_to_active,
    extract_entity,
    PassiveDetectorAgent,
    ContextRetrieverAgent,
    AgentInferenceAgent,
    MystificationClassifierAgent,
    AgentClassifierAgent,
    VerifierAgent,
    AnnotatorAgent,
    DeducibleAgent
)

def run_pipeline():
    # 1. init deducable agent list
    try:
        deducable_agent_map = {}
        with open('D:\mystic\De-mystification\deducable_agents.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if 'verb' in item and 'deduced_agent' in item:
                    verb = item['verb']
                    agent = item['deduced_agent']
                    deducable_agent_map[verb] = agent
            print(f"Loaded deducable agents: {len(deducable_agent_map)} entries\n")
    except FileNotFoundError:
        print("There is no 'deducable_agents' file. We will skip this.\n")
        
    # 2. init PassivePy
    try:
        passivepy = PassivePy.PassivePyAnalyzer(spacy_model = "en_core_web_lg")
        print(f"Loaded PassivePy model: {passivepy}\n")
    except Exception as e:
        print(f"Failed to load PassivePy. {e}\n")
        return
    
    # 3. init API key
    # load_dotenv()
    # api_key = os.environ.get("OPENAI_API_KEY")
    # if not api_key:
    #     print("No 'OPENAI_API_KEY' key found in environment variables.\n")
    #     return

    # 4. init LLM
    try:
        # llm_model = ChatOpenAI(model_name="gpt-4o", temperature=0.1, openai_api_key=api_key)
        llm_model = ChatOllama(model="gemma3:12b", temperature=0.1, base_url="http://localhost:11434")
        print(f"Loaded language model: {llm_model.model}\n")
    except Exception as e:
        print(f"Failed to load language model. {e}\n")
        return
    
    # 5. init agents
    try:
        passive_detector = PassiveDetectorAgent(passivepy_instance=passivepy)
        context_retriever = ContextRetrieverAgent(llm=llm_model, window_size=5)
        deduce_agent = DeducibleAgent(llm=llm_model)
        agent_inferencer = AgentInferenceAgent(llm=llm_model)
        mystification_classifier = MystificationClassifierAgent(llm=llm_model)
        agent_classifier = AgentClassifierAgent(llm=llm_model, passivepy_analyzer=passivepy)
        verifier = VerifierAgent(llm=llm_model)
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
    print("...Running passive detector agent...")
    sentences_dict = passive_detector.run(sentences_dict)
    if not sentences_dict:
        print("No passive sentences detected. Exiting pipeline.\n")
        return
    print("...Running context retrieve agent...")
    sentences_dict = context_retriever.run(sentences_dict)

    print("...Running deducible agent...")
    sentences_dict = deduce_agent.run(sentences_dict, deducible_agent_map=deducable_agent_map)

    print("...Running classify agent...")
    sentences_dict = agent_classifier.run(sentences_dict)

    print("...Running agent inference agent...")
    sentences_dict = agent_inferencer.run(sentences_dict)

    print("...Running index mystification agent...")
    sentences_dict = mystification_classifier.run(sentences_dict)

    print("...Running verification agent...")
    sentences_dict = verifier.run(sentences_dict)

    print("...Running annotator agent...")
    output = annotator.run(sentences_dict)
    with open('output.json', 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"Output is saved to 'output.json'.\n")

    f.close()

def main():
    run_pipeline()

if __name__ == "__main__":
    main()
