#!/bin/sh
LOG=$1; N=$2
A="plus_sdk:P_RESTEAL_ON=1,P_W3_ON=1,P_XJ_ON=1,P_STAB_ON=1,P_STAB_FRAC=0.5,P_XR_ON=1,P_XR_MARGIN=1.5,P_XR_PRIOR=0.7"
B="$A,P_D3_ON=1,P_D3_GATE=1,P_X4_ON=1,P_W4_ON=1"
for fs in "v3 97" "v3 131" "v3adv3 97" "v3adv4 97" "v3advjam 97" "v3mirror 97"; do
  set -- $fs
  python tools/paired.py "$A" "$B" $1 $N $2 2>&1 | grep "B-A" | sed "s/^/seed$2 /" >> $LOG
done
echo DONE >> $LOG
