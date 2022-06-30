#!/bin/sh

for d in ./*.yaml; do
	echo $d
	kge start $d --search.device_pool cuda:1,cuda:2,cuda:3,cuda:5,cuda:6,cuda:7 --search.num_workers 6 --train.subbatch_auto_tune true
done