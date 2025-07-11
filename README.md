# De-mystification
Utilizing pre-trained large language models to demystify passive-voice sentences.

## Set-up
1. Clone the repository
```
git clone https://github.com/mitramir55/PassivePy.git
cd De-mystification/
```
2. Virtual environment
```
conda create -n <environment-name> python=3.10
conda activate <environment-name>
```
3. Installing requirements
```
pip3 install -r requirements.txt
```
4. Set-up environment for LLM <br/>
- For OpenAI's API key (by default), create an `.env` file:
```
OPENAI_API_KEY="YOUR_API_KEY"
```
- If you decide to use a local language model (e.g: Ollama, etc...), define it inside `main.py`:
```
llm_model = ChatOllama(model="gemma3:12b", temperature=0.1, base_url="http://localhost:11434")
```
- One might need to change `base_url` to the localhost of one llm's localhost and the `model` to one need.
## Run
Run the pipeline with
```
python3 main.py
```

## Output
If everything go smoothly, you should have an `output.json` like this:
```
{
    "dummy": [
        {
            "text": "The book was carefully examined by the town's historian, who was fascinated by its intricate designs.",
            "voice_type": "1",
            "context": "In a village encircled by forests, an unusual event occurred as the community prepared for their annual festival. A mysterious book, hidden for generations in the roots of an ancient tree, was discovered. The town's historian, intrigued by the book's intricate designs, examined it closely. The symbols within the book were believed to hold the key to a long-lost treasure. As news of the discovery spread, the village elders decided to embark on a quest to find the treasure. They created a map and entrusted it to a select group of courageous individuals. The following morning, this group of adventurers set out on their journey to uncover the hidden treasure.",
            "agent_status": "explicit",
            "mystification_idx": "1",
            "verb_phrase": "was carefully examined by",
            "agent": "the town's historian"
        },
        {
            "text": "The symbols within the pages were thought to be the key to a forgotten treasure.",
            "voice_type": "2",
            "context": "In a village encircled by forests, an unusual event took place as the community prepared for their annual festival. A mysterious book, hidden for generations within the roots of an ancient tree, was discovered. The town's historian, captivated by the book's intricate designs, examined it and speculated that the symbols inside might lead to a forgotten treasure. This exciting possibility quickly spread among the villagers, prompting the elders to convene and decide to pursue the treasure. They created a map and entrusted it to a select group of courageous individuals. The following morning, these adventurers embarked on their quest, venturing into the woods. However, their journey soon became challenging as a thick fog enveloped them, adding an element of mystery and uncertainty to their expedition.",
            "agent_status": "implied",
            "mystification_idx": "2",
            "verb_phrase": "were thought",
            "agent": "the town's historian"
        }
    ]
}
```
