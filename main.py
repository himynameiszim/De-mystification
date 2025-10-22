import os
import sys
import json
import multiprocessing
from functools import partial
from tqdm import tqdm
# import warnings

# warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# # importing PassivePy
# PassivePySRC_path = "path_toPassivePySrc"
# if PassivePySRC_path not in sys.path:
#     sys.path.append(PassivePySRC_path)

try:
    from PassivePySrc import PassivePy
except ImportError as e:
    print(f"Cannot import PassivePy: {e}")
    sys.exit(1)

# from langchain_community.chat_models import ChatOpenAI # Uncomment if using OpenAI
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

agent = {} # Dictionary to hold all agents

def load_document() -> str:
    """
    Load the deducable agents list from directory (if available) and the corpus from directory.
    """
    # 1. Load deducable agent list (if available)
    try:
        deducable_agent_list_path = input("Enter deducable agents list file path (or press Enter to skip): ").strip()
        deducable_agent_map = {}
        with open(deducable_agent_list_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if 'verb' in item and 'deduced_agent' in item:
                    verb = item['verb']
                    agent = item['deduced_agent']
                    deducable_agent_map[verb] = agent
            print(f"Loaded deducable agents: {len(deducable_agent_map)} entries\n")
    except FileNotFoundError:
        print("There is no 'deducable_agents' file. We will skip this.\n")

    # 2. Load corpus from directory
    corpus_path = input("Enter corpra input directory: ").strip()
    if not os.path.isdir(corpus_path):
        print(f"Invalid directory path: {corpus_path}\n")
        return
    print(f"Reading files from: {corpus_path}")
    sentences_dict = read_txt_files_to_sentences_dict(corpus_path)
    if not sentences_dict:
        print("No text files found or files were empty in the specified directory.")
        return
    print(f"Loaded {len(sentences_dict)} file(s)\n")

    return sentences_dict, deducable_agent_map

def initialize_agent():
    """
    Initialize all necessary components and agents for the pipeline.
    """    
    # 1. Initialize PassivePy
    try:
        passivepy = PassivePy.PassivePyAnalyzer(spacy_model = "en_core_web_lg")
        print(f"Loaded PassivePy model: {passivepy}\n")
    except Exception as e:
        print(f"Failed to load PassivePy. {e}\n")
        return

    # 2. For OpenAI (uncomment if using OpenAI)
    # load_dotenv()
    # api_key = os.environ.get("OPENAI_API_KEY")
    # if not api_key:
    #     print("No 'OPENAI_API_KEY' key found in environment variables.\n")
    #     return

    # 3. Initialize LLM model (adjust if needed)
    try:
        llm_model = ChatOllama(model="qwen2.5:1.5b", temperature=0.1, base_url="http://localhost:11434") # example for Ollama, for openAI, an API key parameter is needed
        print(f"Loaded language model: {llm_model.model}\n")
    except Exception as e:
        print(f"Failed to load language model. {e}\n")
        return

    # 4. Initialize agents
    try:
        agent['passive_detector'] = PassiveDetectorAgent(passivepy_instance=passivepy)
        agent['context_retriever'] = ContextRetrieverAgent(llm=llm_model, window_size=5)
        agent['deduce_agent'] = DeducibleAgent(llm=llm_model)
        agent['agent_inferencer'] = AgentInferenceAgent(llm=llm_model)
        agent['mystification_classifier'] = MystificationClassifierAgent(llm=llm_model)
        agent['agent_classifier'] = AgentClassifierAgent(llm=llm_model, passivepy_analyzer=passivepy)
        agent['verifier'] = VerifierAgent(llm=llm_model)
        agent['annotator'] = AnnotatorAgent()
        print("Loaded all agents.\n")
    except Exception as e:
        print(f"Failed to initialize agents. {e}\n")
        return

def demystify(file_item, deducable_agent_map):
    if not agent:
        initialize_agent()
    
    filename, sentences = file_item
    single_file_dict = {filename: sentences}

    sentences_dict = agent['passive_detector'].run(single_file_dict)
    if not sentences_dict:
        print("No passive sentences to process. Exit now.\n")
        return filename, {}
    
    sentences_dict = agent['context_retriever'].run(sentences_dict)
    sentences_dict = agent['deduce_agent'].run(sentences_dict, deducable_agent_map=deducable_agent_map)
    sentences_dict = agent['agent_inferencer'].run(sentences_dict)
    sentences_dict = agent['mystification_classifier'].run(sentences_dict)
    sentences_dict = agent['agent_classifier'].run(sentences_dict)
    sentences_dict = agent['verifier'].run(sentences_dict)

    return filename, sentences_dict.get(filename, {})

def run_pipeline():
    sentences_dict, deducable_agent_map = load_document()
    num_files = len(sentences_dict)
    num_cores = multiprocessing.cpu_count()
    print(f"Processing with {num_cores} cores...\n")

    tasks = sentences_dict.items()
    agent_func = partial(demystify, deducable_agent_map=deducable_agent_map)

    final_sentences_dict = {}
    with multiprocessing.Pool(processes=num_cores) as pool:
        for filename, processed_sentences in tqdm(pool.imap_unordered(agent_func, tasks), total=num_files, desc="Processing files"):
            if processed_sentences:
                final_sentences_dict[filename] = processed_sentences
    
    print("Done.\n")

    print("...Running annotator...\n")
    annotator = AnnotatorAgent()
    output = annotator.run(final_sentences_dict)
    with open("output.json", 'w', encoding='utf-8') as f:
        f.write(output)
    print("output.json saved.\n")

    f.close()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    run_pipeline()