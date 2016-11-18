#!/bin/bash

if [ "$#" -eq 3 ]; then
  ./halite -d "$1" "python3 $2" "python3 $3" -q
elif [ "$#" -eq 4 ]; then
  ./halite -d "$1" "python3 $2" "python3 $3" "python3 $4" -q
elif [ "$#" -eq 5 ]; then
  ./halite -d "$1" "python3 $2" "python3 $3" "python3 $4" "python3 $5" -q
elif [ "$#" -eq 6 ]; then
  ./halite -d "$1" "python3 $2" "python3 $3" "python3 $4" "python3 $5" "python3 $6" -q
fi
