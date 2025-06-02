class VerifierAgent:
    def __init__(self, llm):
        '''
        TO BE IMPLEMENTED: VerifierAgent is an agent that verifies the mystification classification of a sentence.
        '''
        super().__init__()
        self.chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["sentence", "classification"],
                template=(
                    "Verify the mystification classification for the sentence.\n"
                    "Sentence: {sentence}\n"
                    "Classification: {classification}\n"
                    "If correct, respond with 'yes', otherwise suggest a correction (Level 1-4)."
                ),
            ),
        )

    def run(self, inputs):
        sentence, classification = inputs
        return self.chain.run(sentence=sentence, classification=classification).strip().lower()