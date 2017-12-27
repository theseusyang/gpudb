#!/bin/bash

powa=1
for((i=1;i<=27;i++))
do 

./a.out $powa ``

powa=`expr $powa \* 2`

done


