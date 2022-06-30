import pandas as pd

df = pd.read_csv('all_trials.csv')
idx = df.groupby(['job_id'])['metric'].transform(max) == df['metric']
df_final  = df[idx].copy()
df_final['model'] = df_final.folder.map(lambda x: "-".join(x.split('-')[6:-2]))
df_final = df_final[['job_id', 'model', 'metric', 'fwt_mrr_entity', 'fwt_hits@1_entity', 'fwt_hits@3_entity',
       'fwt_hits@10_entity', 'fwt_mrr_relation', 'fwt_hits@1_relation',
       'fwt_hits@3_relation', 'fwt_hits@10_relation', 'fwt_mrr', 'fwt_hits@1',
       'fwt_hits@3', 'fwt_hits@10']]

print(df_final.head(7))
print(df_final.groupby(['model']).agg(['mean','std']).round(5))
print(df.columns)
df_final.groupby(['model']).agg(['mean','std']).round(5).to_csv('mean_std.csv')
