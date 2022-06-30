#!/bin/sh

for d in ./*.yaml; do
	echo $d
	kge start $d --search.device_pool cuda:7 --search.num_workers 1 --train.subbatch_auto_tune true
done