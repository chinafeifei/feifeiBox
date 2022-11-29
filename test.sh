#!/usr/bin sh

set -e
File_result=$HOME/"$(date +%H%M%S)"_result.txt

j=1
for ((i=0; i<15; i++));
do
	loc="$j"p""
	str=`sed -n $loc a.txt`
	j=`expr $j + 1`
	echo $str
done

#echo `sed -n 1p a.txt`
