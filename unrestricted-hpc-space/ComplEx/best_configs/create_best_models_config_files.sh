#!/bin/bash

dataset=${PWD##*/}
for d in */ ; do
	cd $d
	IFS='/'	read -ra ADDR <<< "$d"
	model=${ADDR[0]}
	kge dump trace . --keysfile /work/dhuynh/Workspaces/Thesis/kge/local/experiments/full_search_hybrid_er_rp/thesis_keys.conf > trace_dump.csv
	trial_folder=$(python ../get_best_trial.py)
	cd $trial_folder
	echo 'Dumping config of best '$model' for '$dataset'...'
	output_file=$model'-config-checkpoint_best.yaml'
	kge dump config checkpoint_best.pt --exclude search ax_search dataset.files job.device > $output_file
	mv $output_file ../../$output_file
	cd ../..
done