#!/bin/sh

for d in ./*.yaml; do
	echo $d
	kge start $d --search.device_pool cuda:0,cuda:1 --search.num_workers 2
done