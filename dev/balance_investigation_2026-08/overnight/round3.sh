#!/bin/bash
cd /home/kuceral4/projects/ostrich
PY=.venv/bin/python
S=/tmp/claude-1000/-home-kuceral4-projects-ostrich/6854194e-bd46-4738-b518-2ec169ef8b40/scratchpad
O=$S/overnight
FB=examples/helhest/helhest_balance_feedback.py

# Arm A: drift curriculum — continue from r2_linwarm with stronger position
# pressure (fall-ambiguity is fixed by wrot=400 + alive-gated best tracking).
timeout 10800 $PY $FB --duration 1.5 --vis headless --exact-bptt \
  --orient-loss quadratic --weight-rot 400 --weight-pos 5 \
  --policy linear --init-policy $O/r2_linwarm.npz \
  --inject-scale 0.3 --lr 0.005 --iterations 400 \
  --save-policy $O/r3_linwarm_wpos5.npz \
  2>&1 | grep -E "^Iter|best|saved" > $O/train_r3_linwarm.log

# Arm B: small-capacity residual, gentler regime.
timeout 10800 $PY $FB --duration 1.5 --vis headless --exact-bptt \
  --orient-loss quadratic --weight-rot 400 --weight-pos 1 \
  --policy residual --hidden 4 \
  --inject-scale 0.2 --lr 0.002 --iterations 400 \
  --save-policy $O/r3_residual_h4.npz \
  2>&1 | grep -E "^Iter|best|saved" > $O/train_r3_residual.log

for cand in "pd $O/pd_base.npz" "r2_linwarm $O/r2_linwarm.npz" \
            "r3_linwarm_wpos5 $O/r3_linwarm_wpos5.npz" \
            "r3_residual_h4 $O/r3_residual_h4.npz"; do
  set -- $cand
  echo "== $1" >> $O/eval3.log
  [ -f $2 ] && timeout 3600 $PY $FB --eval-only --duration 3.0 --vis headless \
    --orient-loss quadratic --weight-rot 400 --weight-pos 1 \
    --init-policy $2 --eval-kicks "0,-0.3,0.3" 2>&1 | grep "^EVAL" >> $O/eval3.log
  [ -f $2 ] && timeout 3600 $PY $FB --eval-only --duration 6.0 --vis headless \
    --orient-loss quadratic --weight-rot 400 --weight-pos 1 \
    --init-policy $2 --eval-kicks "0" 2>&1 | sed 's/^EVAL/EVAL6s/' | grep "^EVAL" >> $O/eval3.log
done
echo ROUND3_DONE
