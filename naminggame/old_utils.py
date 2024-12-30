import os 
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import random 
import nltk
from nltk.corpus import words

### Model Utils 
class LLMAnswerGenerator:
    def __init__(self, model_path, temperature):
        tensor_parallel_size = int(os.environ.get("DEVICES", "1"))
        self.temperature = temperature
        self.sampling_params = SamplingParams(max_tokens=200, temperature=self.temperature)
        self.model = LLM(model=model_path, quantization="awq", dtype="auto")
    
    def get_prompt(self, instruction):
        '''
        Constructing a prompt template with a default system prompt and a dynamic instruction.
        '''
        #B_INST, E_INST = "[INST]", "[/INST]"
        #B_SYS, E_SYS = "<>\n", "\n<>\n\n"
        DEFAULT_SYSTEM_PROMPT = """
        You are an agent with your own language and vocabulary, participating in a word game with several other agents. 
        You will be repeatedly shown the words in your current vocabulary and asked if you would like to add another new word to the vocabulary. 
        Only respond in "yes/no" according to your preferences in the following manner, without any additional tokens. Some examples are given below 
        Question: Your words are: abstain, high, low. Do we add gender to the list?
        Answer: No 
        Question: Your words are: general, finesse. Do we add regular to the list?
        Answer: Yes
        """
        #SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
        SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
        #prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
        prompt_template =  SYSTEM_PROMPT + instruction
        return prompt_template

    def generate_answer(self, prompt_text):
        raw_answer = self.model.generate(prompt_text, sampling_params=self.sampling_params, use_tqdm=False)
        answer = raw_answer[0].outputs[0].text
        return answer

    def deallocate_llm(self):
        import gc
        import torch.cuda
        #import torch.distributed
        destroy_model_parallel()
        del self.model.llm_engine.model_executor.driver_worker
        del self.model # Isn't necessary for releasing memory, but why not
        gc.collect()
        torch.cuda.empty_cache()

### HELPER FUNCTIONS
def generate_vocabulary(vocab_size):
  # Download the nltk english word corpus
  nltk.download('words')
  # Get a list of English words
  english_words = words.words()
  vocabulary = random.sample(english_words, vocab_size)
  # Convert words to lowercase
  vocabulary = [word.lower() for word in vocabulary]
  return vocabulary

# Generate vocabulary for agents with a seed
def create_agent_vocabulary(seed, num_agents, max_words, vocab):
    # Initialize an empty dictionary to store the v[i] values
    vocab_dict = {}
    random.seed(seed)
    # Populate the v[i] dictionary with unique random choices
    for i in range(num_agents):
        vocab_dict[i] = list(set(random.choice(vocab) for _ in range(max_words)))
    return vocab_dict

### VALUES TO BE PLOTTED
def get_number_of_words(state, num_steps):
    # For Number of words Nw(t) plot
    number_of_words = [(kk, len([word for sublist in state[kk].values() for word in sublist])) for kk in range(num_steps)]
    return number_of_words

def get_number_of_unique_words(state, num_steps):
    number_unique_words = [(kk, len(set(word for sublist in state[kk].values() for word in sublist))) for kk in range(num_steps)]
    return number_unique_words 

### RUN MULTIPLE EXPERIMENTS
def get_temperatures(experiment_params):
    temperatures = [0.00]
    for param in experiment_params:
        if "temperature" in param:
            temperatures = param["temperature"]
    return temperatures

### GET YES/NO FROM UNPREDICTABLE OUTPUT
def find_first_yes_or_no(text):
    # Convert text to lowercase to handle case insensitivity
    text = text.lower()
    
    # Find the first occurrence of "yes" or "no"
    yes_index = text.find("yes")
    no_index = text.find("no")
    
    # Determine which comes first
    if yes_index == -1 and no_index == -1:
        return "no"
    elif yes_index == -1:
        return "yes"
    elif no_index == -1:
        return "no"
