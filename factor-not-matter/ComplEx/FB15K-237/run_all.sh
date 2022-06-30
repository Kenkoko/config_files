#!/bin/sh

for d in ./*.yaml; do
	echo $d
	kge start $d --search.device_pool cuda:1,cuda:1,cuda:1,cuda:1,cuda:2,cuda:2,cuda:2,cuda:2,cuda:3,cuda:3 --search.num_workers 10 --train.subbatch_auto_tune true
done
