#!/bin/sh
while [ "$(grep -c DONE data/fp.log 2>/dev/null)" -lt 2 ]; do sleep 20; done
PKG="P_VSUS_ON=1,P_D3_ON=0,P_VSUS_GATE=0.26,P_W3_FIX=1,P_RS_SIZE=2.8,P_XR_MAX_BB=20"
rm -f data/chk_oj.log
sh tools/ab_r4.sh data/chk_oj.log 4000 309 "$PKG" "$PKG,P_CHK_ON=1" v4
sh tools/ab_r4.sh data/chk_oj.log 4000 309 "$PKG" "$PKG,P_OJ_ON=1" v4
cat data/fp.log data/chk_oj.log
