#!/bin/sh

for d in ./*.yaml; do
	echo $d
	for i in {1..5}; do 
		kge start $d --train.subbatch_auto_tune true
	done
	
done