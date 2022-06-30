import copy
from collections import Counter

import pandas as pd
import torch
from kge.job.eval import EvaluationJob
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from functools import reduce
import numpy as np

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# download link for this checkpoint given under results above
PATH = "/work/dhuynh/Workspaces/Thesis/kge/local/experiments/error_analysis"


def analyze_relation(path, eval_type, suffix=''):
    
    # load model
    checkpoint = load_checkpoint(f'{PATH}/best_models/{path}/checkpoint_best.pt')
    model = KgeModel.create_from(checkpoint)
    
    # determine device
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    # load model to device
    model.to(device)
    model.config.set('job.device', device)
    model.config.set('eval.batch_size', 4)

    # determine evaluation job
    model.config.set("eval.split", eval_type)
    data = model.dataset.split(eval_type)
    num_rel = model.dataset.num_relations()
    
    
    
    # evaluating
    eval = EvaluationJob.create(config=model.config, 
                                dataset=model.dataset, 
                                model=model)
    eval._prepare()
    eval._run()
    ranks_filt_test = eval.all_ranks_filt_test
    for key in ranks_filt_test.keys():
        ranks_filt_test[key] = ranks_filt_test[key].cpu().detach().numpy()
    name_subjects = model.dataset.entity_strings(data[:, 0])
    name_relation = model.dataset.relation_strings(data[:, 1])
    name_objects = model.dataset.entity_strings(data[:, 2])
    relation_type = [model.dataset.index("relation_types")[id] for id in data[:, 1]]

    # predicting
    subject_list = []
    subject_score_list = []
    relation_list = []
    object_list = []
    object_score_list = []
    all_pairs = []
    for idx, batch_coords in enumerate(eval.loader):
        # print('batch:', idx)
        batch = batch_coords[0].to(device)
        s, p, o = batch[:, 0], batch[:, 1], batch[:, 2]
        
        pairs = zip(list(model.dataset.entity_strings(s)), 
                    list(model.dataset.relation_strings(p)), 
                    list(model.dataset.entity_strings(o))) 
        
        all_pairs += [" | ".join(pair) for pair in list(pairs)]
        
        # scores of all predicate for (s,?,o)
        scores = model.score_so(s, o) 
        # filtering  
        so_index = model.dataset.index(f"{eval_type}_so_to_p")
        p_idx = torch.stack([torch.range(0, len(batch) - 1, dtype=torch.long).to(device), p], dim=0).t()
        coords = so_index.get_all(batch[:, [0, 2]].detach().cpu()).to(device)
        # print('coords:', coords)
        for x in p_idx:
            coords = coords[coords.eq(x).all(dim=1).logical_not()]
        coords = coords.long()
        # print('coords:', coords)
        # print(f'scores - size ({scores.size()}):', scores)
        if coords.nelement() != 0:
            scores[coords[:, 0], coords[:, 1]] = -float("Inf")
        
        # print('scores:', scores)
        predicted_p = torch.topk(scores, 3)
        predicted_p_indices = predicted_p.indices
        # print('predicted_p:', predicted_p)
        # index_reciprocal_relation = (predicted_p >= num_rel).nonzero().squeeze().detach().cpu().numpy()
        
        # print('predicted_p:', predicted_p)
        index_reciprocal_relation = torch.nonzero(predicted_p.indices >= num_rel)
        if index_reciprocal_relation.nelement() != 0:
            predicted_p_indices[index_reciprocal_relation[:, 0], index_reciprocal_relation[:, 1]] = predicted_p_indices[index_reciprocal_relation[:, 0], index_reciprocal_relation[:, 1]] - num_rel
        relation_names = list(model.dataset.relation_strings(predicted_p_indices))

        # relation_names = np.vstack(relation_names)
        # print(f'relation_names {type(relation_names)}:', relation_names)
        
        # if index_reciprocal_relation.nelement() != 0:
        #     relation_names[index_reciprocal_relation[:, 0].detach().cpu(), index_reciprocal_relation[:, 1].detach().cpu()] = relation_names[index_reciprocal_relation[:, 0].detach().cpu(), index_reciprocal_relation[:, 1].detach().cpu()] + ' (reciprocal)'
        # if len(index_reciprocal_relation) > 0:
        #     for idx in index_reciprocal_relation:
        #         relation_names[idx] = relation_names[idx] + " (reciprocal)"
        relation_list += relation_names
        
        # print('relation_names:', relation_names)
        # raise Exception('stop')
        
        # scores of all predicate for (s,p,_)
        scores = model.score_sp(s, p)
        predicted_o = torch.topk(scores, 5)
        predicted_o_indices = predicted_o.indices
        predicted_o_score =  predicted_o.values.detach().cpu().numpy().tolist()
        object_list += list(model.dataset.entity_strings(predicted_o_indices))
        # print(predicted_o_score)
        object_score_list += predicted_o_score
        
        # scores of all predicate for (_,p,o)
        scores = model.score_po(p, o)
        predicted_s = torch.topk(scores, 5)
        predicted_s_indices = predicted_s.indices
        predicted_s_score = predicted_s.values.detach().cpu().numpy().tolist()
        subject_list += list(model.dataset.entity_strings(predicted_s_indices))
        subject_score_list += predicted_s_score

    # create dataframe
    output = {
        'name_subjects': name_subjects,
        'name_relation': name_relation,
        'name_objects': name_objects,
        'relation_type': relation_type,
        'triplet': all_pairs,
        f'predicted_subject_{suffix}': subject_list,
        f'predicted_subject_score_{suffix}': subject_score_list,
        f'predicted_relation_{suffix}': relation_list,
        f'predicted_object_{suffix}': object_list,
        f'predicted_object_scores_{suffix}': object_score_list,
    }
    
    # rename keys
    for key in ranks_filt_test.keys():
        new_key = f'{key}_{suffix}'
        ranks_filt_test[new_key] = ranks_filt_test.pop(key)
    
    output = {**output, **ranks_filt_test}
    

    output_df = pd.DataFrame.from_dict(output).reset_index().rename(columns={'index': 'id'})
    print(output_df.head(3)[['triplet', f'predicted_subject_{suffix}', f'predicted_relation_{suffix}', f'predicted_object_{suffix}']])
    return output_df

hyperparameters_name = 'ICLR20'
model = 'ComplEx'

# evel_dataset = 'valid'
# # evel_dataset = 'train'

for eval_type in ['train', 'valid']:
    for dataset in ['FB237', 'WNRR']:
        for model_selection in ['ER', 'Hybrid']:
            dfList = []
            for train_type in ['standard', 'hybrid']:
                tmp_df = analyze_relation(f'{hyperparameters_name}/{model}/{dataset}/{model_selection}_{train_type}', 
                                        eval_type, 
                                        suffix=train_type)
                
                dfList.append(tmp_df)

            key_columns = ['id', 'name_subjects', 'name_relation', 'name_objects', 'relation_type', 'triplet']

            df = reduce(lambda df1, df2: pd.merge(df1, df2, on=key_columns), dfList)
            df = df[key_columns + sorted(list(set(df.columns).difference(set(key_columns))))]
            # print(df)
            # print(df.columns)
            df.to_csv(f'{PATH}/results/{hyperparameters_name}/{model}/{dataset}/{dataset.lower()}_{model_selection.lower()}_error_analysis_ranking_{eval_type}.csv', 
                    index=False)






