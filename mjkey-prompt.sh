#!/bin/bash

F="$HOME/.mujoco/mjkey.txt"
if [ "$(cat $F|wc -l)" == "0" ]
then
	echo "Replace the contents of this file with your Mujoco Key" > $F
	vim $F
fi

echo "Mujoco key is present"
cat $F | grep -E "Issued|Expires"
