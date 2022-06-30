import pandas as pd

df = pd.read_csv('all_trials.csv')
idx = df.groupby(['folder'])['metric'].transform(max) == df['metric']

df[idx].drop_duplicates().to_csv('best_models.csv', index=False)

# print(df.columns)
