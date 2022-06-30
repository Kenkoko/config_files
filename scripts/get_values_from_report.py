
import re
import os
import pandas as pd

PATH = '/work/dhuynh/Workspaces/Thesis/kge/local/experiments/full_search_hybrid_er_rp/ComplEx/FB15K-237/ER/'

metric = {
    'fwt_mrr_entity'      :'mean_reciprocal_rank_filtered_with_test_entity',
    'fwt_hits@1_entity'   :'hits_at_1_filtered_with_test_entity',
    'fwt_hits@3_entity'   :'hits_at_3_filtered_with_test_entity',
    'fwt_hits@10_entity'  :'hits_at_10_filtered_with_test_entity',
    'fwt_mrr_relation'    :'mean_reciprocal_rank_filtered_with_test_relation',
    'fwt_hits@1_relation' :'hits_at_1_filtered_with_test_relation',
    'fwt_hits@3_relation' :'hits_at_3_filtered_with_test_relation',
    'fwt_hits@10_relation':'hits_at_10_filtered_with_test_relation',
    'fwt_mrr'             :'mean_reciprocal_rank_filtered_with_test',
    'fwt_hits@1'          :'hits_at_1_filtered_with_test',
    'fwt_hits@3'          :'hits_at_3_filtered_with_test',
    'fwt_hits@10'         :'hits_at_10_filtered_with_test',
}


results = {}
## get all txt files in directory
for file in os.listdir(PATH):
    if file.endswith(".txt"):
        path_file = os.path.join(PATH, file)
        model = "-".join(file.split('-')[5:-2])
        
        f = open(path_file, "r")
        report = f.read()
        
        output = {}
        for key in metric:
            value = re.findall(rf'{metric[key]}: (.+)\n', report)[0]
            output[key] = value
        results[model] = output

pd.DataFrame.from_dict(results, orient='index').to_csv(f'{PATH}/new_report.csv')

