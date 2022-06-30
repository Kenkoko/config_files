#!/bin/sh

for d in ./*.yaml; do
	echo $d
	kge start $d --search.device_pool cuda:2,cuda:3 --search.num_workers 2 --train.subbatch_size 512 --entity_ranking.chunk_size 100000
done