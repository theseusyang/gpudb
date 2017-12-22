#!/bin/bash

powa=2
for((i=1;i<=29;i++))
do 
powa=`expr $powa \* 2`

./a.out $powa 
done


