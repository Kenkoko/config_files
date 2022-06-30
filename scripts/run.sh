#!/bin/bash

./create_dump.sh /work/dhuynh/Workspaces/Thesis/kge/local/experiments/full_search_hybrid_er_rp/thesis_keys.conf 
./merge_csv.sh
python get_best.py 