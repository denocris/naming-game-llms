import random
import csv
import time
from tqdm import tqdm

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
def run_llm_naming_game(num_agents, num_steps, vocabulary, answer_generator, print_answers=True):
    # Initialize dictionaries
    p = {agent: [] for agent in range(num_agents)}  # Dictionary of speakers and listeners
    
    # List to store success indicator
    succes_indicator = [0] * num_steps 
    # Pi and Phi
    # Pi ("yes" | in inventory): when entry is 1 the chosen word by i is part of the inventory of j. It is a true positive since the agent replies "yes"
    pi_true_positive = [0] * num_steps 
    # Phi ("yes" | not in inventory): when entry is 1 the chosen word by i is NOT part of the inventory of j. It is a false positive since the agent replies "yes
    phi_false_positive = [0] * num_steps 
    # Pi ("no" | in inventory): corresponds to false negative: the agent replies "no" but the chosen word by i was in the inventory of j
    pi_false_negative = [0] * num_steps
    # Pi ("no" | not in inventory): corresponds to true negative: the agent replies "no" and in fact the chosen word by i was NOT in the inventory of j
    phi_true_negative = [0] * num_steps 


    number_of_words = {step: [] for step in range(num_steps)} 
    number_of_diff_words = {step: [] for step in range(num_steps)}  

    for k in tqdm(range(num_steps)):
        #print(f"Step no : {k}")
        #time.sleep(5)
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

        instruction = f"Your words are: {p[j]}. Do we add {chosen_word} to the list?"
        prompt_text = answer_generator.get_prompt(instruction)
        answer = answer_generator.generate_answer(prompt_text)
        #print(prompt_text[0]['content']+prompt_text[1]['content'])
        #print("instruction:", instruction)
        #print("answer:", answer)
        # Extract yes/no from the output  
        #answer = find_first_yes_or_no(answer)
        #if print_answers: print("answer NG:", answer)

        # if the word is in both in speaker and listener
        if answer == "yes":
            #print("1")
            p[i] = [chosen_word] #speaker
            p[j] = [chosen_word] #listener
            succes_indicator[k] = 1
            if chosen_word in p[j]:
                pi_true_positive[k]=1 
            else:
                phi_false_positive[k]=1
        elif answer == "no":
            #print("2")
            p[j] = list(set(p[j] + [chosen_word]))
            succes_indicator[k] = 0
            if chosen_word in p[j]:
                pi_false_negative[k]=1 
            else:
                phi_true_negative[k]=1
        else:
            #print("3")
            "If anything else, consider it as a no"
            p[j] = list(set(p[j] + [chosen_word]))
            succes_indicator[k] = 0

        state_k = {agent: p[agent] for agent in range(num_agents)}
        number_of_words[k] = len([word for sublist in state_k.values() for word in sublist])
        number_of_diff_words[k] = len(set([word for sublist in state_k.values() for word in sublist]))
    return number_of_words, number_of_diff_words, succes_indicator, pi_true_positive, phi_false_positive, pi_false_negative, phi_true_negative

