import copy
from collections import Counter

import pandas as pd
import torch
from kge.job.eval import EvaluationJob
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from functools import reduce

# download link for this checkpoint given under results above
PATH = "/work/dhuynh/Workspaces/Thesis/kge/local/experiments/error_analysis"


def analyze_relation(path, eval_type, suffix='', top=10, ):
    
    # load model
    checkpoint = load_checkpoint(f'{PATH}/best_models/{path}/checkpoint_best.pt')
    model = KgeModel.create_from(checkpoint)
    
    # determine device
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # load model to device
    model.to(device)
    model.config.set('job.device', device)
    
    # determine evaluation job
    model.config.set("eval.split", eval_type)
    valid_data = model.dataset.split(eval_type)
    
    # get list of available relations
    relation_ids = valid_data[:, 1]
    flatten_relation_ids = torch.flatten(relation_ids).numpy()
    
   
    # perfom evaluation
    output = {}
    for id, count in Counter(flatten_relation_ids).most_common(top):
        
        ## get name of relation
        name_relation = model.dataset.relation_strings(torch.Tensor([id,]).long())[0]
        
        # # get all triplets with this relation
        # index = torch.flatten((relation_ids == id).nonzero())
        # print(id)
        # assert index.size(0) == count
        # selected_tripets = torch.index_select(model.dataset.split(eval_type), 0, index)
        new_dataset = copy.deepcopy(model.dataset)
        new_dataset._triples[eval_type] = valid_data[valid_data[:, 1] == id]
        assert valid_data[valid_data[:, 1] == id].size(0) == count
        # evaluating
        eval = EvaluationJob.create(config=model.config, 
                                    dataset=new_dataset, 
                                    model=model)
        eval._prepare()
        eval._run()
        
        # append results
        result = {'id': id, 'count':valid_data[valid_data[:, 1] == id].size(0), 'relation_type': model.dataset.index("relation_types")[id]}
        name = "_" + suffix if suffix != '' else ""
        result = {**result, 
                  **eval._compute_metrics(eval.hists_filt_test['relation'], suffix="_filtered_with_test_relation" + name),
                  }
        output[name_relation] = result

    output_df = pd.DataFrame.from_dict(output, orient='index').reset_index().rename(columns={'index': 'relation'})
    return output_df

hyperparameters_name = 'ICLR20'
model = 'ComplEx'
top = None
for eval_type in ['train', 'valid']:
    for dataset in ['FB237', 'WNRR']:
        for model_selection in ['ER', 'Hybrid']:
            dfList = []
            for type in ['hybrid', 'standard']:
                tmp_df = analyze_relation(f'{hyperparameters_name}/{model}/{dataset}/{model_selection}_{type}', 
                                        eval_type, 
                                        suffix=type,
                                        top=top)
                dfList.append(tmp_df)

            key_columns = ['relation', 'id', 'count', 'relation_type']

            df = reduce(lambda df1, df2: pd.merge(df1, df2, on=key_columns), dfList)
            df = df[key_columns + sorted(df.columns[len(key_columns):], reverse=True)]
            df.to_csv(f'{PATH}/results/{hyperparameters_name}/{model}/{dataset}/{dataset.lower()}_{model_selection.lower()}_error_analysis_metric_{eval_type}.csv', 
                    index=False)




