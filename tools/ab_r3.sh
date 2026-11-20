#!/bin/sh
# usage: tools/ab_r3.sh LOG TABLES SEED "variant" field1 field2 ...   (base = round-3 submission config)
LOG=$1; N=$2; SD=$3; V=$4; shift 4
A="plus_sdk:P_RESTEAL_ON=1,P_W3_ON=1,P_XJ_ON=1,P_STAB_ON=1,P_STAB_FRAC=0.5,P_XR_ON=1,P_XR_MARGIN=1.5,P_XR_PRIOR=0.7,P_D3_ON=1,P_D3_GATE=1,P_X4_ON=1,P_W4_ON=1"
for f in "$@"; do
  python tools/paired.py "$A" "$A,$V" $f $N $SD 2>&1 | grep "B-A" | sed "s/^/[$V seed$SD] /" >> $LOG
done
echo DONE >> $LOG
