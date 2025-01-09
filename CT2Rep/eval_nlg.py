import os
import evaluate
import pandas as pd
from tqdm import tqdm

# Load metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")


data_root = '/data/houbb/PyCharm_Deployment/CT2Rep/CT2Rep/results/ct2rep_baseline'

gts = pd.read_csv(os.path.join(data_root, '40gts.csv'), header=None)
res = pd.read_csv(os.path.join(data_root, '40res.csv'), header=None)
gts[0] = gts[0].replace('\s*\[SEP\]\s*|\s*\[PAD\]\s*', ' ', regex=True).str.strip()
res[0] = res[0].replace('\s*\[SEP\]\s*|\s*\[PAD\]\s*', ' ', regex=True).str.strip()

a=1

# Iterate through references and candidates
results = []
for idx in tqdm(range(1505)):
    pred_text = res.iloc[idx].values[0]
    true_text = gts.iloc[idx].values[0]

    bleu_result = bleu.compute(predictions=[pred_text], references=[true_text])
    rouge_result = rouge.compute(predictions=[pred_text], references=[true_text])
    meteor_result = meteor.compute(predictions=[pred_text], references=[true_text])
    results.append({'idx': idx, **bleu_result, **rouge_result, **meteor_result})

pd.set_option('display.max_columns', None)

results_df = pd.DataFrame(results)
precisions_expanded = pd.DataFrame(results_df['precisions'].to_list(), index=results_df.index)

print(results_df.describe())
print(precisions_expanded.describe())

a=1
