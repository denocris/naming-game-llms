import os 
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer
import random 
import nltk
from nltk.corpus import words
import ollama

### Model Utils 
class LLMAnswerGenerator:
    def __init__(self, model_path, temperature):
        self.checkpoint = model_path 
        self.temperature = temperature
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        #self.model = T5ForConditionalGeneration.from_pretrained(self.checkpoint)   
        #self.tokenizer = T5Tokenizer.from_pretrained(self.checkpoint) 
        
    def get_prompt(self, instruction):
        '''
        Constructing a prompt template with a default system prompt and a dynamic instruction.
        '''
        DEFAULT_SYSTEM_PROMPT = """
        You are an agent with your own language and vocabulary. You can and must reply with yes or no. 
        """
        SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
        prompt_template =  SYSTEM_PROMPT + instruction
        return prompt_template

    def generate_answer(self, prompt_text):
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        outputs = self.model.generate(**inputs, temperature=self.temperature, max_new_tokens=1)
        # [-1] is necessary since model.generate() outputs prompt + answer
        answer_tmp = self.tokenizer.decode(outputs[0][-1])
        # T5 cleaning
        #cleaned_answer = answer.replace("<pad>", "").replace("</s>", "").lower().replace(' ', '')
        # Llama-3.2-1B cleaning 
        answer = answer_tmp.replace("<|begin_of_text|>", "").lower()
        # Extracting "yes" or "no" using string manipulation
        #print("answer LLM:", answer)
        return answer
    
### Model Utils 
class GroqLLMAnswerGenerator:
    def __init__(self, model_path, groq_client, temperature):
        self.checkpoint = model_path 
        self.temperature = temperature
        self.client = groq_client
        
    def get_prompt(self, instruction):
        '''
        Constructing a prompt template with a default system prompt and a dynamic instruction.
        '''
        DEFAULT_SYSTEM_PROMPT = "You are an agent with your own language and vocabulary. You can and must reply with yes or no. "
        #SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
        prompt_template = [
                    # Set an optional system message. This sets the behavior of the
                    # assistant and can be used to provide specific instructions for
                    # how it should behave throughout the conversation.
                    {
                        "role": "system",
                        "content": DEFAULT_SYSTEM_PROMPT
                    },
                    # Set a user message for the assistant to respond to.
                    {
                        "role": "user",
                        "content": instruction,
                    }]
        return prompt_template

    def generate_answer(self, prompt_text):
        response = self.client.chat.completions.create(model=self.checkpoint,
                                    messages=prompt_text,
                                    max_tokens=1,
                                    temperature=self.temperature)
        cleaned_answer = response.choices[0].message.content.strip().lower()
        return cleaned_answer
    
class OllamaLLMAnswerGenerator:
    def __init__(self, model_path, temperature):
        self.checkpoint = model_path
        self.temperature = temperature
        #self.client = ollama_client
        
    def get_prompt(self, instruction):
        '''
        Constructing a prompt template with a default system prompt and a dynamic instruction.
        '''
        DEFAULT_SYSTEM_PROMPT = "You are an agent with your own language and vocabulary. You can and must reply with yes or no. "
        #SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
        prompt_template = [
                    # Set an optional system message. This sets the behavior of the
                    # assistant and can be used to provide specific instructions for
                    # how it should behave throughout the conversation.
                    {
                        "role": "system",
                        "content": DEFAULT_SYSTEM_PROMPT
                    },
                    # Set a user message for the assistant to respond to.
                    {
                        "role": "user",
                        "content": instruction,
                    }]
        return prompt_template

    def generate_answer(self, prompt_text):
        prompt_text = prompt_text[0]['content']+prompt_text[1]['content']
        response = ollama.generate(model=self.checkpoint, 
                         prompt=prompt_text, 
                         options={"temperature": self.temperature, "num_predict": 1})
        cleaned_answer = response['response'].strip().lower()
        return cleaned_answer

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
def ____deprecated_find_first_yes_or_no(text):
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

