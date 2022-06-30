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


def analyze_relation(path, df_relations, list_all_relations, suffix=''):
    
    # load model
    checkpoint = load_checkpoint(f'{PATH}/best_models/{path}/checkpoint_best.pt')
    model = KgeModel.create_from(checkpoint)
    
    # determine device
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # load model to device
    model.to(device)
    model.config.set('job.device', device)
    
    # load data
    model.dataset.relation_strings(torch.Tensor([1,]).long())
    model.dataset.entity_strings(torch.Tensor([1,]).long())

    meta_relation = model.dataset._meta['relation_strings']
    meta_entity = model.dataset._meta['entity_strings']
    
    output = []
    print(df_relations.columns)

    list_so = list(set(df_relations.so_pair.values))
    id_subjects = []
    id_objects = []
    for so_pair in list_so:
        name_subject = so_pair.split(', _ ,')[0]
        name_object = so_pair.split(', _ ,')[1]
        if name_subject in meta_entity:
            id_subjects.append(meta_entity.index(name_subject))
        else:
            id_subjects.append(8) # Warning: could not find 8 ids in map entity_strings; filling with None.
        if name_object in meta_entity:
            id_objects.append(meta_entity.index(name_object))
        else:
            id_objects.append(8) # Warning: could not find 8 ids in map entity_strings; filling with None.

    # convert ids to tensor
    
    num_batch = 256
    id_subjects_chunks = [id_subjects[x:x+num_batch] for x in range(0, len(id_subjects), num_batch)]
    id_objects_chunks = [id_objects[x:x+num_batch] for x in range(0, len(id_objects), num_batch)]

    
    ids = list(zip(id_subjects_chunks, id_objects_chunks))
    relation_list = []
    for id_subjects, id_objects in ids:

        id_subjects = torch.tensor(id_subjects).long().to(device)
        id_objects = torch.tensor(id_objects).long().to(device)

        scores = model.score_so(id_subjects, id_objects)  # scores of all predicate for (s,?,o)
        p = torch.argmax(scores, dim=-1)
        
        # reciprocal model
        index_reciprocal_relation = (p >= len(meta_relation)).nonzero().squeeze().detach().cpu().numpy()
        p[p >= len(meta_relation)] = p[p >= len(meta_relation)] - len(meta_relation)
        relation_names = list(model.dataset.relation_strings(p))
        
        if len(index_reciprocal_relation) > 0:
            for idx in index_reciprocal_relation:
                relation_names[idx] = relation_names[idx] + " (reciprocal)"

        relation_list += relation_names
    
    df_predicted_relation = pd.DataFrame({'so_pair': list_so, f'predicted_relation_{suffix}': relation_list})
    df_relations = df_relations.merge(df_predicted_relation, how='left', on=['so_pair'])


    list_relation = list(set(df_relations.name_relation.values))
    output = []
    for group_relations in list_relation:
        ids_relations = []
        for relation in group_relations:
            ids_relations.append(meta_relation.index(relation))
        
        ids = torch.Tensor(ids_relations).long().to(device)
        
        embedder = model.get_p_embedder()
        embedding_vectors = embedder._embeddings(ids)
        
        
        output.append({
            'name_relation': group_relations,
            'relation_type': tuple([model.dataset.index("relation_types")[id] for id in ids_relations]),
            'counts': (list_all_relations.count(group_relations[0]), list_all_relations.count(group_relations[1])),
            'is_similar': model.dataset.index("relation_types")[ids_relations[0]] == model.dataset.index("relation_types")[ids_relations[1]],
            f'cosine_sim_{suffix}': torch.nn.functional.cosine_similarity(embedding_vectors[0], embedding_vectors[1], dim=0).item(),
            f'euclidean_dist_{suffix}': torch.nn.functional.pairwise_distance(embedding_vectors[0], embedding_vectors[1], p=2).item(),
        })
    df = pd.DataFrame(output)
    if 'relation_type' in df_relations.columns:
        return df_relations.merge(df, how='left', on=['name_relation', 'relation_type', 'counts', 'is_similar'])
    return df_relations.merge(df, how='left', on=['name_relation'])

    
    


hyperparameters_name = 'ICLR20'
model = 'ComplEx'

# for eval_type in ['train', 'valid']:
for dataset in ['FB237', 'WNRR']:
    all_df = []
    for model_selection in ['ER', 'Hybrid']:
        # load ranking results
        
        df_list = []
        for eval_type in ['train', 'valid']:
            df = pd.read_csv(f'{PATH}/results/{hyperparameters_name}/{model}/{dataset}/{dataset.lower()}_{model_selection.lower()}_error_analysis_ranking_{eval_type}.csv')
            df['eval_type'] = eval_type
            df_list.append(df)
        
        df_ranking = pd.concat(df_list)
        # getting relations having same subject and object
        so_more_one_rel = df_ranking[df_ranking.duplicated(subset=['name_subjects', 'name_objects', 'eval_type'],keep=False)].sort_values(by=['name_subjects', 'name_objects', 'eval_type'])
        so_more_one_rel['so_pair'] = so_more_one_rel['name_subjects'] + ', _ ,' + so_more_one_rel['name_objects']
        
        
        def pair_wise_list_relations(list_relation):
            # deduplicate list of lists
            # list_relation = [list(x) for x in set(tuple(x) for x in list_relation)]
            if len(list_relation) == 1:
                return [tuple(list_relation)]
            # combinations of the list in a group of 2
            output = list(itertools.combinations(list_relation, 2))
            output = [tuple(sorted(x)) for x in output]
            # print(output)
            return list(output)
        
        # get list of relations having same subject and object
        similar_relations = so_more_one_rel.groupby(['so_pair', 'eval_type'])['name_relation'].apply(lambda x: pair_wise_list_relations(list(set(x))))#.drop_duplicates().values
        similar_relations = similar_relations.reset_index().explode('name_relation').drop_duplicates()

        # remove list of relations having same only 1 relation
        similar_relations = similar_relations[similar_relations.name_relation.apply(lambda x: len(x) > 1)]

        # find ranking of relations having same subject and object
            # indexing ranking df
        df_ranking['fact'] = df_ranking.apply(lambda x: f'{x.name_subjects}:{x.name_relation}:{x.name_objects}', axis=1)
        df_ranking = df_ranking.drop_duplicates(subset=['fact'])
        df_ranking = df_ranking.set_index('fact')
        df_ranking_dict = df_ranking.to_dict('index')
        
        
        
        def get_ranking(row, type_training):
            name_subject = row.so_pair.split(', _ ,')[0]
            name_object = row.so_pair.split(', _ ,')[1]
            fact_1 = f'{name_subject}:{row.name_relation[0]}:{name_object}'
            fact_2 = f'{name_subject}:{row.name_relation[1]}:{name_object}'
            ranking_1 = df_ranking_dict[fact_1][f'so_to_p_{type_training}']
            ranking_2 = df_ranking_dict[fact_2][f'so_to_p_{type_training}']
            return tuple([ranking_1, ranking_2])
        # print('-----------------------------------------------------')
        # print('similar_relations before adding ranking')
        # print(similar_relations)
        similar_relations['so_to_p_standard'] = similar_relations.apply(lambda row: get_ranking(row, 'standard'), axis=1)
        similar_relations['so_to_p_hybrid'] = similar_relations.apply(lambda row: get_ranking(row, 'hybrid'), axis=1)

        # print('similar_relations after adding ranking')
        # print(similar_relations.head())
        # print('-----------------------------------------------------')

        for type in ['hybrid', 'standard']:
            similar_relations = analyze_relation(
                f'{hyperparameters_name}/{model}/{dataset}/{model_selection}_{type}', # load model
                suffix=type, # distinguish between hybrid and standard
                df_relations=similar_relations, # relations to analyze
                list_all_relations=list(df_ranking.name_relation.values) # all relations
            )
        
        # print(similar_relations.columns)
        # raise Exception('stop')
            

        # print(similar_relations.columns)
        key_columns = ['so_pair', 'name_relation', 'eval_type', 'relation_type', 'counts', 'is_similar']
        similar_relations = similar_relations[key_columns + sorted(list(set(similar_relations.columns).difference(set(key_columns))), reverse=True)]
        print('similar_relations after adding similarity')
        print(similar_relations.columns)
        print(similar_relations[key_columns].head())

        all_df.append(similar_relations)

    df = reduce(lambda df1, df2: pd.merge(df1, df2, on=key_columns, suffixes=('_ER', '_Hybrid')), all_df)
    df = df[key_columns + sorted(df.columns[len(key_columns):], reverse=True)].sort_values(by=['so_pair'])
    print(df)
    df.to_csv(f'{PATH}/results/{hyperparameters_name}/{model}/{dataset}/{dataset.lower()}_error_analysis_relations_in_similar_so_all.csv', 
            index=False)






