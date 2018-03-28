#!/bin/bash

powa=128
for((i=7;i<=28;i++))
do 
taskset -c 63 ./smallstore  67108864   67108864 $powa   

powa=`expr $powa \* 2`

done

