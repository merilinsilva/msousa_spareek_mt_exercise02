'''
Merilin Silva
Shubhi Pareek

Here please use the command: python tools/pytorch-examples/word_language_model/connect_logs.py 'FULL PATH TO MODELS_DROPOUT FOLDER'
'''

######Imports########
import pandas as pd
import os
import re
import sys
####################

# We need the full directory for later
dropout_dir = sys.argv[1]
# We first need to get all the dropout logs
files = os.listdir(dropout_dir)

df_combined = pd.DataFrame()

for file in files:
    # We need to have the full path
    filepath = os.path.join(dropout_dir, file)

    try:
        df = pd.read_csv(filepath, sep='\t')

        dropout_value = re.search(r'dropout_(0\.\d+)', file)
        # This is in case there is no match
        if not dropout_value:
            continue

        dropout = dropout_value.group(1)
        df = df[['epoch', 'valid_perplexity']]

        # We found this nice function that allows us to rename a column :)
        df.rename(columns={'valid_perplexity': f'Dropout {dropout}'}, inplace=True)

        if df_combined.empty:
            df_combined["Valid. perplexity"] = df['epoch']

        df_combined[f'Dropout {dropout}'] = df[f'Dropout {dropout}']

    except Exception as e:
        print(f"{file}: {e}")

# Finally, we saved our results
output_path = os.path.join(dropout_dir, 'combined_dropout_results.tsv')
df_combined.to_csv(output_path, sep='\t', index=False)
