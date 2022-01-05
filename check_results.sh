#!/bin/sh


nb_row=`wc -l < $1`
nb_row=`echo "($nb_row)*(6.000000)" | bc`
echo "nb_row = $nb_row"
cumul=0.0

cur_line=1
paste -d@ $1 $2 | while IFS="@" read -r f1 f2
do
    ds=0.0
    arr1=$(echo $f1 | tr ' ' " ")
    arr2=$(echo $f2 | tr ' ' " ")
    for i in $arr1
    do
        ds=`echo "($ds)+($i)" | bc`
    done
    for i in $arr2
    do
        ds=`echo "($ds)-($i)" | bc`
    done
    
    echo "`echo "100.0*($cur_line)/($nb_row)" | bc`%"
    cur_line=`echo "($cur_line)+(1.0)" | bc`
    
    cumul=`echo "($cumul)+sqrt(($ds)*($ds))" | bc`
    mean=`echo "100.0*($cumul) / ($nb_row)" | bc`
    echo "($cumul)/..($nb_row) = $mean"
    echo "mean_difference*100 = $mean"
    
done
