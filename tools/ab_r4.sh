#!/bin/sh
# usage: tools/ab_r4.sh LOG TABLES SEED "A-extra" "B-extra" field1 field2 ...   (base = round-3 config)
LOG=$1; N=$2; SD=$3; XA=$4; XB=$5; shift 5
BASE="plus_sdk:P_RESTEAL_ON=1,P_W3_ON=1,P_XJ_ON=1,P_STAB_ON=1,P_STAB_FRAC=0.5,P_XR_ON=1,P_XR_MARGIN=1.5,P_XR_PRIOR=0.7,P_D3_ON=1,P_D3_GATE=1,P_X4_ON=1,P_W4_ON=1"
A="$BASE"; [ -n "$XA" ] && A="$BASE,$XA"
B="$BASE,$XB"
for f in "$@"; do
  python tools/paired.py "$A" "$B" $f $N $SD 2>&1 | grep "B-A" | sed "s/^/[$XB seed$SD] /" >> $LOG
done
echo DONE >> $LOG
