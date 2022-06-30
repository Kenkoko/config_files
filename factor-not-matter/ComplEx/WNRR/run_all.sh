#!/bin/sh

for d in ./*.yaml; do
	echo $d
	kge start $d --search.device_pool cuda:5,cuda:5,cuda:5,cuda:5,cuda:5,cuda:6,cuda:6,cuda:6,cuda:6,cuda:6 --search.num_workers 10 --train.subbatch_auto_tune true
done