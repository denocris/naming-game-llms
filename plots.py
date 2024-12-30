import matplotlib.pyplot as plt 
import pandas as pd


# Read the data
df = pd.read_json("./results/num_of_words_llmama-3.2-1B_log_T0.75.json")

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['word_count'], label='Number of Words')
plt.xlabel('Time Steps')
plt.ylabel('Number of Words')
plt.title('Evolution of Total Number of Words Over Time')
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig('./images/num_words_evolution.png')
plt.show()



# Read the data for different words
df_diff = pd.read_json("./results/num_of_diff_words_llmama-3.2-1B_log_T0.75.json")

# Create the plot for different words
plt.figure(figsize=(10, 6))
plt.plot(df_diff.index, df_diff['diff_word_count'], label='Number of Different Words')
plt.xlabel('Time Steps')
plt.ylabel('Number of Different Words')
plt.title('Evolution of Number of Different Words Over Time')
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig('./images/num_diff_words_evolution.png')
plt.show()

success_data_path = "./results/success_llmama-3.2-1B_log_T0.75.csv"
df_success = pd.read_csv(success_data_path)
success_indicator = df_success.values[::, 1]

T = len(success_indicator)

CS = [0] * (T+1)
for i in range(1, T):
    CS[i + 1] = CS[i] + success_indicator[i]

# Create cs list
cs = [(i, CS[i]) for i in range(T)]

# Define δ
δ = 300

# Calculate dsDeterministic
dsNG= [(i, (cs[i + δ][1] - cs[i][1]) / δ) for i in range(T - δ)]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(*zip(*dsNG))
plt.title("Success rate S(t)")
plt.xlabel("t")
plt.ylabel("Success rate")
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.show()

