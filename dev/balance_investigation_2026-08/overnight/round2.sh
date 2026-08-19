#!/bin/bash
cd /home/kuceral4/projects/ostrich
PY=.venv/bin/python
S=/tmp/claude-1000/-home-kuceral4-projects-ostrich/6854194e-bd46-4738-b518-2ec169ef8b40/scratchpad
O=$S/overnight
FB=examples/helhest/helhest_balance_feedback.py

train2() { name=$1; shift
  timeout 10800 $PY $FB --duration 1.5 --vis headless --exact-bptt \
    --orient-loss quadratic --weight-rot 400 --weight-pos 1 \
    --inject-scale 0.3 --lr 0.005 --save-policy $O/$name.npz "$@" \
    2>&1 | grep -E "^Iter|best|saved" > $O/train_$name.log
  tail -1 $O/train_$name.log; }

train2 r2_linwarm --policy linear --init-policy $O/pd_base.npz --iterations 400
train2 r2_residual --policy residual --iterations 400
train2 r2_residual_kick --policy residual --kick-std 0.2 --iterations 400

for cand in "pd $O/pd_base.npz" "r2_linwarm $O/r2_linwarm.npz" \
            "r2_residual $O/r2_residual.npz" "r2_residual_kick $O/r2_residual_kick.npz"; do
  set -- $cand
  echo "== $1 (3s)" >> $O/eval2.log
  [ -f $2 ] && timeout 3600 $PY $FB --eval-only --duration 3.0 --vis headless \
    --orient-loss quadratic --weight-rot 400 --weight-pos 1 \
    --init-policy $2 --eval-kicks "0,-0.3,0.3" 2>&1 | grep "^EVAL" >> $O/eval2.log
done
echo ROUND2_DONE
