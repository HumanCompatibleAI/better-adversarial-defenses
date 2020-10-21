#!/bin/bash

F="$HOME/.mujoco/mjkey.txt"

test_key()
{
	cd $HOME/.mujoco
	key_ret=$(yes | $HOME/.mujoco/mujoco200/bin/testxml $HOME/.mujoco/mujoco200/model/humanoid.xml)
	key_res=$?
}

test_key
while [ "$key_res" != "0" ]
do
	echo > emptyfile
	echo "Asking for key"
	dialog --title "Input your Mujoco key here" --editbox emptyfile 20 100 2>$F
	rm emptyfile
        test_key
	clear
	if [ "$key_res" != "0" ]
	then
		echo "Mujoco responded with (press Enter to continue)"
		echo "$key_ret"
		read
	fi
done

echo "Mujoco key is present now"
cat $F | grep -E "Issued|Expires"
