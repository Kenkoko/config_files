#!/bin/sh

for d in ./*.yaml; do
	echo $d
	kge start $d --search.device_pool cuda:1,cuda:2,cuda:3,cuda:6 --search.num_workers 4 --train.subbatch_auto_tune true --entity_ranking.chunk_size 15000
done