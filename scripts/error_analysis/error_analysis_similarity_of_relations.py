import copy
from collections import Counter

import pandas as pd
import torch
from kge.job.eval import EvaluationJob
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from functools import reduce
import itertools


# download link for this checkpoint given under results above
PATH = "/work/dhuynh/Workspaces/Thesis/kge/local/experiments/error_analysis"


def analyze_relation(path, suffix=''):
    
    # load model
    checkpoint = load_checkpoint(f'{PATH}/best_models/{path}/checkpoint_best.pt')
    model = KgeModel.create_from(checkpoint)
    
    # determine device
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # load model to device
    model.to(device)
    model.config.set('job.device', device)
    
    num_relations = model.dataset.num_relations()
    print(num_relations)
    list_permutation = list(itertools.combinations(range(num_relations), 2))
    print(len(list_permutation))
    # print(list_permutation)
    output = []
    embedder = model.get_p_embedder()
    
    for i in range(len(list_permutation)):

        relation_id_1 = torch.Tensor([list_permutation[i][0],]).long()
        relation_id_2 = torch.Tensor([list_permutation[i][1],]).long()
        
        relation_string_1 = model.dataset.relation_strings(relation_id_1)[0]
        relation_string_2 = model.dataset.relation_strings(relation_id_2)[0]
        
        ids = torch.Tensor([relation_id_1, relation_id_2]).long().to(device)
        embedding_vectors = embedder._embeddings(ids)
        
        output.append({
            'pair_relation': f'{relation_string_1}, {relation_string_2}',
            f'cosine_sim_{suffix}': torch.nn.functional.cosine_similarity(embedding_vectors[0], embedding_vectors[1], dim=0).item(),
            # f'euclidean_sim_{suffix}': torch.nn.functional.pairwise_distance(embedding_vectors[0], embedding_vectors[1], p=2).item(),
        })
    df = pd.DataFrame(output)
    return df


hyperparameters_name = 'ICLR20'
model = 'ComplEx'

# for eval_type in ['train', 'valid']:


for dataset in ['FB237', 'WNRR']:
    all_df = []
    key_columns = ['pair_relation']
    for model_selection in ['ER', 'Hybrid']:
        for training_objective in ['hybrid', 'standard']:
            similarity_df = analyze_relation(
                f'{hyperparameters_name}/{model}/{dataset}/{model_selection}_{training_objective}', # load model
                suffix=f'{training_objective}_{model_selection}', # distinguish between hybrid and standard
            )
            all_df.append(similarity_df)

    df = reduce(lambda df1, df2: pd.merge(df1, df2, on=key_columns), all_df)
    df = df[key_columns + sorted(df.columns[len(key_columns):], reverse=True)].sort_values(by=['pair_relation'])
    print(df)
    df.to_csv(f'{PATH}/results/{hyperparameters_name}/{model}/{dataset}/{dataset.lower()}_error_analysis_relation_similarity.csv', 
            index=False)






