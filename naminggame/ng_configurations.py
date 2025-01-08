import random
import csv
import time
from tqdm import tqdm
from naminggame.utils import find_first_yes_or_no

############ DETERMINISTIC NAMING GAME #############
def run_deterministic_naming_game(num_agents, num_steps, vocabulary, PRINT=False):
    # Initialize dictionaries
    p = {agent: [] for agent in range(num_agents)}  # Dictionary of speakers and listeners
    succes_indicator = [0] * num_steps  # List to store success indicator
    number_of_words = {step: [] for step in range(num_steps)} 
    number_of_diff_words = {step: [] for step in range(num_steps)}  

    for k in tqdm(range(num_steps)):
        i = random.randint(0, num_agents - 1) #speaker
        j = random.randint(0, num_agents - 1) #listener
        if PRINT: print(k, i, j)
        if PRINT: print('p before:', p)

        #if speaker is empty
        if not p[i]:
            w = random.choice(vocabulary)
            if PRINT: print(w)
            p[i] = [w]

        #if listener is empty
        if not p[j]:
            w = random.choice(vocabulary)
            if PRINT: print(w)
            p[j] = [w]

        chosen_word = random.choice(p[i])
        if PRINT: print(chosen_word)

        # if the word is in both in speaker and listener
        if set([chosen_word]).intersection(p[j]):
            p[i] = [chosen_word] #speaker
            p[j] = [chosen_word] #listener
            succes_indicator[k] = 1
        else:
            p[j] = p[j] + [chosen_word]
            succes_indicator[k] = 0

        if PRINT: print('p    :', p)
        state_k = {agent: p[agent] for agent in range(num_agents)}
        number_of_words[k] = len([word for sublist in state_k.values() for word in sublist])
        number_of_diff_words[k] = len(set([word for sublist in state_k.values() for word in sublist]))
        if PRINT: print('state: ', state_k)
    return number_of_words, number_of_diff_words, succes_indicator

############ LLM NAMING GAME #############
def run_llm_naming_game(num_agents, num_steps, vocabulary, answer_generator):
    # Initialize dictionaries
    p = {agent: [] for agent in range(num_agents)}  # Dictionary of speakers and listeners
    succes_indicator = [0] * num_steps  # List to store success indicator
    number_of_words = {step: [] for step in range(num_steps)} 
    number_of_diff_words = {step: [] for step in range(num_steps)}  

    for k in tqdm(range(num_steps)):
        #print(f"Step no : {k}")
        if k % 100 == 0 and k != 0:  # Exclude the first step (0)
            print("Pausing for 15 seconds...")
            time.sleep(15)
        i = random.randint(0, num_agents - 1) #speaker
        j = random.randint(0, num_agents - 1) #listener

        #if speaker is empty
        if not p[i]:
            w = random.choice(vocabulary)
            p[i] = [w]

        #if listener is empty
        if not p[j]:
            w = random.choice(vocabulary)
            p[j] = [w]

        chosen_word = random.choice(p[i])

        instruction = f"Your words are: {p[i]}. Do we add {chosen_word} to the list?"
        prompt_text = answer_generator.get_prompt(instruction)
        answer = answer_generator.generate_answer(prompt_text)
        # Extract yes/no from the output  
        answer = find_first_yes_or_no(answer)

        # if the word is in both in speaker and listener
        if answer == "yes":
            p[i] = [chosen_word] #speaker
            p[j] = [chosen_word] #listener
            succes_indicator[k] = 1
        elif answer == "no":
            p[j] = p[j] + [chosen_word]
            succes_indicator[k] = 0

        state_k = {agent: p[agent] for agent in range(num_agents)}
        number_of_words[k] = len([word for sublist in state_k.values() for word in sublist])
        number_of_diff_words[k] = len(set([word for sublist in state_k.values() for word in sublist]))
    return number_of_words, number_of_diff_words, succes_indicator

############ LLM NAMING GAME #############
def run_llm_naming_game_old(num_agents, num_steps, vocabulary, model_name, answer_generator):
    # Initialize dictionaries
    p = {agent: [] for agent in range(num_agents)}  # Dictionary of speakers and listeners
    succes_indicator = [0] * num_steps  # List to store success indicator
    state = {step: [] for step in range(num_steps)}   # Dictionary to store system state
    version_number = random.randint(1, 10000)
    csv_filename = f"{model_name}_game_state_log_v{version_number}.csv"

    with open(csv_filename, mode='w', newline='') as csvfile:
        fieldnames = ['step'] + [f'agent_{i}' for i in range(num_agents)]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for k in tqdm(range(num_steps)):
            #print(f"Step no : {k}")
            i = random.randint(0, num_agents - 1) #speaker
            j = random.randint(0, num_agents - 1) #listener

            #if speaker is empty
            if not p[i]:
                w = random.choice(vocabulary)
                p[i] = [w]

            #if listener is empty
            if not p[j]:
                w = random.choice(vocabulary)
                p[j] = [w]

            chosen_word = random.choice(p[i])

            instruction = f"Your words are: {p[i]}. Do we add {chosen_word} to the list?"
            prompt_text = answer_generator.get_prompt(instruction)
            answer = answer_generator.generate_answer(prompt_text)

            # if the word is in both in speaker and listener
            if answer == "Yes":
                p[i] = [chosen_word] #speaker
                p[j] = [chosen_word] #listener
                succes_indicator[k] = 1
            else:
                p[j] = p[j] + [chosen_word]
                succes_indicator[k] = 0

            state[k] = {agent: p[agent] for agent in range(num_agents)}

            # Write the current state to the CSV file
            row = {'step': k}
            for agent in range(num_agents):
                #row[f'agent_{agent}'] = ' '.join(p[agent])
                row[f'agent_{agent}'] = p[agent]
            writer.writerow(row)

    return state, succes_indicator
