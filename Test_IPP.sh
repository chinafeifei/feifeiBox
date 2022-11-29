#!/bin/bash

set -e

WORKDIR=$(find ~  -name 'applications.industrial.machine-vision.computer-vision-optimization-toolkit-pv_rc1' | head -n 1)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/ipp/latest/lib/intel64/tl/tbb
source /opt/intel/oneapi/setvars.sh
source /opt/intel/openvino_2021.4.752/bin/setupvars.sh
File_run=$HOME/run_reult.txt
File_result=$HOME/"$(date +%H%M%S)"_result.txt

# Run Sobel Sample
cd $WORKDIR/Reference-Optimized-Implementation/Optimized-OpenCV-Operators/"CPU Implementation"/SobelGradient
mkdir build
cd build
cmake ..
make VERBOSE=1
k=14
j=6
sum_ipp=0
sum_cv=0

make run-hypot >> $File_run
loc_str1="$j"p""
loc_str2="$k"p""
str=`sed -n $loc_str1 $File_run`
array_opencv=($str)
max_cv=${array_opencv[4]}
min_cv=${array_opencv[4]}
str2=`sed -n $loc_str2 $File_run`
array_ipp=($str2)
max_ipp=${array_ipp[6]}
min_ipp=${array_ipp[6]}


for ((i=0; i<10; i++));
do
	make run-hypot >> $File_run
	loc_str1="$j"p""
        loc_str2="$k"p""
	
	str=`sed -n $loc_str1 $File_run`
	array_opencv=($str)
	if ((${array_opencv[4]} > $max_cv))
	then
		max_cv=${array_opencv[4]}
	elif ((${array_opencv[4]} < $min_cv))
	then
		min_cv=${array_opencv[4]}	
	fi

	sum_cv=`expr $sum_cv + ${array_opencv[4]}`

	str2=`sed -n $loc_str2 $File_run`
	array_ipp=($str2)
	if ((${array_ipp[6]} > $max_ipp))
        then
                max_ipp=${array_ipp[6]}
	elif ((${array_ipp[6]} < $min_ipp))
	then
                min_ipp=${array_ipp[6]}
        fi

	sum_ipp=`expr $sum_ipp + ${array_ipp[6]}`

        j=`expr $j + 20`
	k=`expr $k + 20`
done

printf "%-12s %-10s %-10s %-10s %-10s %-10s %-10s\n" Filter OpenCV_Min OpenCV_Ave OpenCV_Max CVOI_Min CVOI_Ave CVOI_Max >> $File_result
printf "%-12s %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f\n" Sobel_hypot $min_cv `expr $sum_cv / 10` $max_cv $min_ipp `expr $sum_ipp / 10` $max_ipp >> $File_result

rm $File_run
cd ..
rm -r build


# Run Gauss Sample
cd $WORKDIR/Reference-Optimized-Implementation/Optimized-OpenCV-Operators/"CPU Implementation"/GaussianBlur
mkdir build
cd build
cmake ..
make VERBOSE=1

k=51
j=12
sum_ipp=0.00
sum_cv=0.00
make run-blur >> $File_run
loc_str1="$j"p""
loc_str2="$k"p""

str=`sed -n $loc_str1 $File_run`
array_opencv=($str)
max_cv=${array_opencv[9]}
min_cv=${array_opencv[9]}

str2=`sed -n $loc_str2 $File_run`
array_ipp=($str2)
max_ipp=${array_ipp[9]}
min_ipp=${array_ipp[9]}

j=`expr $j + 54`
k=`expr $k + 54`

for ((i=0; i<10; i++));
do
        make run-blur >> $File_run
        loc_str1="$j"p""
        loc_str2="$k"p""

        str=`sed -n $loc_str1 $File_run`
        array_opencv=($str)
        a=${array_opencv[9]}
        #echo $a
        if ((`echo " ${array_opencv[9]} > $max_cv" | bc` == 1))
        then
                max_cv=${array_opencv[9]}
        elif ((`echo "${array_opencv[9]} < $min_cv" | bc` == 1))
        then
                min_cv=${array_opencv[9]}
        fi
        sum_cv="$(echo $sum_cv + $a | bc)"

        str2=`sed -n $loc_str2 $File_run`
        array_ipp=($str2)
        b=${array_ipp[9]}
        #echo $b
        if ((`echo "${array_ipp[9]} > $max_ipp" | bc` == 1))
        then
                max_ipp=${array_ipp[9]}
        elif ((`echo "${array_ipp[9]} < $min_ipp" | bc` == 1))
        then
                min_ipp=${array_ipp[9]}
        fi

        sum_ipp="$(echo $sum_ipp + $b | bc)"

        j=`expr $j + 54`
        k=`expr $k + 54`
done

ave_cv=`echo "scale=2;$sum_cv / 10" | bc`
ave_ipp=`echo "scale=2;$sum_ipp / 10" | bc`

printf "%-12s %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f\n" Gauss_21x21 $min_cv $ave_cv $max_cv $min_ipp $ave_ipp $max_ipp >> $File_result

rm $File_run
cd ..
rm -r build


# Run MorphOpen Sample
cd $WORKDIR/Reference-Optimized-Implementation/Optimized-OpenCV-Operators/"CPU Implementation"/MorphOpen
mkdir build
cd build
cmake ..
make VERBOSE=1

k=6
j=3
sum_ipp=0.00
sum_cv=0.00
make run-open-3 >> $File_run
loc_str1="$j"p""
loc_str2="$k"p""

str=`sed -n $loc_str1 $File_run`
array_opencv=($str)
str_cv=${array_opencv[2]}
arr_cv=(${str_cv//:/ })
max_cv=${arr_cv[1]}
min_cv=${arr_cv[1]}

str2=`sed -n $loc_str2 $File_run`
array_ipp=($str2)
str_ipp=${array_ipp[3]}
arr_ipp=(${str_ipp//:/ })
max_ipp=${arr_ipp[1]}
min_ipp=${arr_ipp[1]}

j=`expr $j + 8`
k=`expr $k + 8`

for ((i=0; i<10; i++));
do
        make run-open-3 >> $File_run
        loc_str1="$j"p""
        loc_str2="$k"p""

        str=`sed -n $loc_str1 $File_run`
        array_opencv=($str)
        str_cv=${array_opencv[2]}
        arr_cv=(${str_cv//:/ })
	a=${arr_cv[1]}
        #echo $a
        if ((`echo " $a > $max_cv" | bc` == 1))
        then
                max_cv=$a
        elif ((`echo "$a < $min_cv" | bc` == 1))
        then
                min_cv=$a
        fi
        sum_cv="$(echo $sum_cv + $a | bc)"

        str2=`sed -n $loc_str2 $File_run`
        array_ipp=($str2)
	str_ipp=${array_ipp[3]}
	arr_ipp=(${str_ipp//:/ })
        b=${arr_ipp[1]}
        #echo $b
        if ((`echo "$b > $max_ipp" | bc` == 1))
        then
                max_ipp=$b
        elif ((`echo "$b < $min_ipp" | bc` == 1))
        then
                min_ipp=$b
        fi

        sum_ipp="$(echo $sum_ipp + $b | bc)"

        j=`expr $j + 8`
        k=`expr $k + 8`
done

ave_cv=`echo "scale=2;$sum_cv / 10" | bc`
ave_ipp=`echo "scale=2;$sum_ipp / 10" | bc`

printf "%-12s %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f\n" MorphOpen3x3 $min_cv $ave_cv $max_cv $min_ipp $ave_ipp $max_ipp >> $File_result

rm $File_run
cd ..
rm -r build


# Run MeanBlur Sample
cd $WORKDIR/Reference-Optimized-Implementation/Optimized-OpenCV-Operators/"CPU Implementation"/MeanBlur
mkdir build
cd build
cmake ..
make VERBOSE=1

k=11
j=4
sum_ipp=0.00
sum_cv=0.00
make run-blur >> $File_run
loc_str1="$j"p""
loc_str2="$k"p""

str=`sed -n $loc_str1 $File_run`
array_opencv=($str)
max_cv=${array_opencv[5]}
min_cv=${array_opencv[5]}

str2=`sed -n $loc_str2 $File_run`
array_ipp=($str2)
max_ipp=${array_ipp[8]}
min_ipp=${array_ipp[8]}

j=`expr $j + 12`
k=`expr $k + 12`

for ((i=0; i<10; i++));
do
        make run-blur >> $File_run
        loc_str1="$j"p""
        loc_str2="$k"p""

        str=`sed -n $loc_str1 $File_run`
        array_opencv=($str)
        a=${array_opencv[5]}
        #echo $a
        if ((`echo " $a > $max_cv" | bc` == 1))
        then
                max_cv=$a
        elif ((`echo "$a < $min_cv" | bc` == 1))
        then
                min_cv=$a
        fi
        sum_cv="$(echo $sum_cv + $a | bc)"

        str2=`sed -n $loc_str2 $File_run`
        array_ipp=($str2)
        b=${array_ipp[8]}
        #echo $b
        if ((`echo "$b > $max_ipp" | bc` == 1))
        then
                max_ipp=$b
        elif ((`echo "$b < $min_ipp" | bc` == 1))
        then
                min_ipp=$b
        fi

        sum_ipp="$(echo $sum_ipp + $b | bc)"

        j=`expr $j + 12`
        k=`expr $k + 12`
done

ave_cv=`echo "scale=2;$sum_cv / 10" | bc`
ave_ipp=`echo "scale=2;$sum_ipp / 10" | bc`

printf "%-12s %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f\n" MeanBlur $min_cv $ave_cv $max_cv $min_ipp $ave_ipp $max_ipp >> $File_result

rm $File_run
cd ..
rm -r build

