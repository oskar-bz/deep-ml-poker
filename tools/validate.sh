#!/bin/sh
LOG=$1; N=$2; SD=$3
A="plus_sdk:P_RESTEAL_ON=1,P_W3_ON=1,P_XJ_ON=1,P_STAB_ON=1,P_STAB_FRAC=0.5,P_XR_ON=1,P_XR_MARGIN=1.5,P_XR_PRIOR=0.7"
B="$A,P_D3_ON=1,P_X4_ON=1,P_W4_ON=1"
for f in v3 v3adv3 v3adv4 v3advjam v3advcall v3advraise v2final; do
  python tools/paired.py "$A" "$B" $f $N $SD 2>&1 | grep "B-A" >> $LOG
done
echo DONE >> $LOG
