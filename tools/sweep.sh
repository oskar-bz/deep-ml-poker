#!/bin/sh
# usage: tools/sweep.sh LOG FIELD TABLES SEED variant1 variant2 ...
LOG=$1; F=$2; N=$3; SD=$4; shift 4
D="plus_sdk:P_RESTEAL_ON=1,P_W3_ON=1,P_XJ_ON=1,P_STAB_ON=1,P_STAB_FRAC=0.5,P_XR_ON=1,P_XR_MARGIN=1.5,P_XR_PRIOR=0.7"
for v in "$@"; do
  r=$(python tools/paired.py "$D" "$D,$v" $F $N $SD 2>&1 | grep "B-A")
  echo "$v :: $r" >> $LOG
done
echo DONE >> $LOG
