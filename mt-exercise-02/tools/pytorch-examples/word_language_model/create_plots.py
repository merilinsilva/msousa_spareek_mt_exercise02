'''
Merilin Silva
Shubhi Pareek

Here, please use the full path to the combined perplexity table as the second argument: python mt-exercise-02/tools/pytorch-examples/word_language_model/create_plots.py 'PATH'
'''

####Imports######
import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
import os
#################

path_perplexities = sys.argv[1]
df = pd.read_csv(path_perplexities, sep='\t')

# We separate the data into training perplexity and validation perplexity
training_df = df[df['Valid. perplexity'].str.contains('Training', case=False)].copy()
validation_df = df[df['Valid. perplexity'].str.contains('Validation', case=False)].copy()

# Next we extract the epoch number
def extract_num(string:str) -> int:
    num = re.search(r'\d+', string)
    return int(num.group())

training_df['epoch'] = training_df['Valid. perplexity'].apply(extract_num)
validation_df['epoch'] = validation_df['Valid. perplexity'].apply(extract_num)

# Next we need to drop the previous column name
training_df.drop(columns=['Valid. perplexity'], inplace=True)
validation_df.drop(columns=['Valid. perplexity'], inplace=True)

# We set epoch as x-axis
training_df.set_index('epoch', inplace=True)
validation_df.set_index('epoch', inplace=True)

# Next we create a folder to store the plots
os.mkdir('plots')

####### Training
plt.figure(figsize=(10, 6))
for col in training_df.columns:
    plt.plot(training_df.index, training_df[col], label=col)
plt.title('Training Perplexity by Dropout Rate')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/perplexity_train.png')

###### Validation
plt.figure(figsize=(10, 6))
for col in validation_df.columns:
    plt.plot(validation_df.index, validation_df[col], label=col)
plt.title('Validation Perplexity by Dropout Rate')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/perplexity_validation.png')
