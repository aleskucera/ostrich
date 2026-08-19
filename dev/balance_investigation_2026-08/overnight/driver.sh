#!/bin/bash
cd /home/kuceral4/projects/ostrich
PY=.venv/bin/python
S=/tmp/claude-1000/-home-kuceral4-projects-ostrich/6854194e-bd46-4738-b518-2ec169ef8b40/scratchpad
O=$S/overnight
FB=examples/helhest/helhest_balance_feedback.py

# Stage 0: wait for the in-flight MLP training to release the GPU.
while pgrep -f "helhest_balance_feedback.*--policy mlp" >/dev/null; do sleep 30; done
echo "stage0 done: previous training finished" | tee -a $O/progress.log

# Warm-start policy file: the stabilizing hand-PD as a linear policy.
$PY -c "
import numpy as np
W = np.zeros((3, 4)); W[:, 1] = 40.0; W[:, 2] = 0.3
np.savez('$O/pd_base.npz', kind='linear', W=W)"

train() { name=$1; shift
  echo \"train $name: $@\" >> $O/progress.log
  timeout 10800 $PY $FB --duration 3.0 --vis headless --exact-bptt \
    --orient-loss quadratic --weight-pos 10 --save-policy $O/$name.npz "$@" \
    2>&1 | grep -E "^Iter|best|saved" > $O/train_$name.log
  tail -1 $O/train_$name.log >> $O/progress.log; }

# Stage 1: linear warm-started from PD.
train linwarm --policy linear --init-policy $O/pd_base.npz --iterations 400 --lr 0.02
# Stage 2: residual MLP (starts at PD behavior, learns corrections).
train residual --policy residual --iterations 400 --lr 0.02
# Stage 3: robust residual (random pitch-rate kicks during training).
train residual_robust --policy residual --kick-std 0.25 --iterations 400 --lr 0.02

# Stage 4: uniform evaluation of every candidate.
evalp() { name=$1; init=$2; dur=$3; kicks=$4
  echo "== $name (dur=$dur)" >> $O/eval.log
  timeout 3600 $PY $FB --eval-only --duration $dur --vis headless \
    --orient-loss quadratic --weight-pos 10 --init-policy $init \
    --eval-kicks $kicks 2>&1 | grep -E "^EVAL" >> $O/eval.log; }

for cand in "pd $O/pd_base.npz" "linear_cold $S/pol_linear.npz" \
            "mlp_cold $S/pol_mlp.npz" "linwarm $O/linwarm.npz" \
            "residual $O/residual.npz" "residual_robust $O/residual_robust.npz"; do
  set -- $cand
  [ -f $2 ] && evalp $1 $2 3.0 "0,-0.2,0.2,-0.4,0.4"
  [ -f $2 ] && evalp $1 $2 6.0 "0"
done
echo OVERNIGHT_DONE | tee -a $O/progress.log
