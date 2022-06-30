#!/bin/bash

./create_dump.sh /data/dhuynh/Thesis/kge/local/experiments/AKBC-reproducibility/thesis_keys.conf
./merge_csv.sh
python get_mean.py