import torch 
import numpy as np
import random 
import pandas as pd
import os
from groq import Groq

from naminggame.utils import get_number_of_words, get_number_of_unique_words, get_temperatures
from naminggame.utils import LLMAnswerGenerator, GroqLLMAnswerGenerator, generate_vocabulary, create_agent_vocabulary
from naminggame.ng_configurations import run_deterministic_naming_game, run_llm_naming_game

import matplotlib.pyplot as plt


import yaml
############ LOAD CONFIGURATION PARAMETERS #############
print("Starting experiment ....")
# Load the config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access general parameters
num_agents = config['num_agents']
num_steps = config['num_steps']
max_words = config['max_words']
seed = config['seed']
vocab_size = config['vocab_size']
deterministic = config['deterministic']

# Print the parameters to verify
print(f"Number of agents: {num_agents}")
print(f"Number of steps: {num_steps}")
print(f"Max words: {max_words}")

############ INITIALIZE RESULTS DIRECTORY #############
if not os.path.exists('results'):
    os.makedirs('results')

############ INITIALIZE SEED #############
print(f"Initialized with seed : {seed}.")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed) 

############ GENERATE AGENT VOCABULARY #############
print("Generating agent vocabulary ...")
vocabulary = generate_vocabulary(vocab_size)

groq_api = config['groq_api']
if groq_api==True:
    groq_key=os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=groq_key)

if deterministic: 
    ############ RUN DETERMINISTIC NAMING GAME #############
    for experiment in config['experiments']:

        print("Run deterministic naming game ...")
        number_of_words, number_of_diff_words, succes_indicator = run_deterministic_naming_game(num_agents = num_agents, num_steps = num_steps, vocabulary = vocabulary)
        #experiment_name = experiment["name"]
        model_name = experiment["name"]
        number_of_words_filename = f"num_of_words_{model_name}_game_state.json"
        number_of_diff_words_filename = f"num_of_diff_words_{model_name}_game_state.json"
        success_filename = f"success_{model_name}_game_state.csv"
        #df_state = pd.DataFrame.from_dict(state)
        df_success = pd.DataFrame.from_dict(succes_indicator)
        df_number_of_words = pd.DataFrame.from_dict(number_of_words, orient='index', columns=['word_count'])
        df_number_of_diff_words = pd.DataFrame.from_dict(number_of_diff_words, orient='index', columns=['diff_word_count'])
        df_number_of_words.to_json("./results/" + number_of_words_filename) 
        df_number_of_diff_words.to_json("./results/" + number_of_diff_words_filename) 
        df_success.to_csv("./results/" + success_filename) 

else: 
    ############ RUN A SET OF LLM EXPERIMENTS FOR DIFFERENT MODELS #############
    for experiment in config['experiments']:
        # Access model parameters
        experiment_name = experiment["name"]
        model_path = experiment['model_parameters'][0]['model_path']
        model_name = experiment['model_parameters'][1]['model_name']
        temperatures = get_temperatures(experiment['model_parameters'])
        
        for temperature in temperatures:
            ############ RUN GAME FOR DIFFERENT TEMPERATURE SETTINGS #############
            print("Running LLM naming game with parameters ... ")
            print(f"Experiment name: {experiment_name}")
            print(f"Model path: {model_path}")
            print(f"Temperature: {temperature}")
    
            # Instantiate the LLM Answer Generator
            if groq_api==True:
                answer_generator = GroqLLMAnswerGenerator(model_path, client, temperature)
            else:
                answer_generator = LLMAnswerGenerator(model_path, temperature)
            number_of_words, number_of_diff_words, succes_indicator = run_llm_naming_game(num_agents = num_agents, num_steps = num_steps, model_name = model_name, answer_generator=answer_generator, vocabulary = vocabulary)
            # Save data
            number_of_words_filename = f"num_of_words_{experiment_name}_{model_name}_log_T{temperature}.json"
            number_of_diff_words_filename = f"num_of_diff_words_{experiment_name}_{model_name}_log_T{temperature}.json"
            success_filename = f"success_{experiment_name}_{model_name}_log_T{temperature}.csv"

            #df_state = pd.DataFrame.from_dict(state)
            df_success = pd.DataFrame.from_dict(succes_indicator)
            df_number_of_words = pd.DataFrame.from_dict(number_of_words, orient='index', columns=['word_count'])
            df_number_of_diff_words = pd.DataFrame.from_dict(number_of_diff_words, orient='index', columns=['diff_word_count'])
            df_number_of_words.to_json("./results/" + number_of_words_filename) 
            df_number_of_diff_words.to_json("./results/" + number_of_diff_words_filename) 
            df_success.to_csv("./results/" + success_filename) 
    



            
