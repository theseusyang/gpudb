#!/bin/bash

powa=32
for((i=5;i<=25;i++))
do 

./a.out 268435456 268435456 $powa ``

powa=`expr $powa \* 2`

done


