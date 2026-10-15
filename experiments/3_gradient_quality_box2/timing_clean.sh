#!/bin/bash
# EXCLUSIVE timing: strictly sequential, single GPU, nothing else running.
cd /local/kuceral4/projects/ostrich/experiments/3_gradient_quality_box2 || exit 1
export CUDA_VISIBLE_DEVICES=1
PM=/local/kuceral4/projects/ostrich/.venv-mjx/bin/python
PW=/local/kuceral4/projects/ostrich/.venv/bin/python
for n in 5 15; do
  echo "[$(date +%H:%M:%S)] MJX 2ms, $n iters"
  $PM optimize_mjx.py --dt 0.002 --lr 0.3 --num-trials 1 --iterations $n \
      --save results/tim_mjx_$n.json > results/tim_mjx_$n.log 2>&1
done
for n in 5 15; do
  echo "[$(date +%H:%M:%S)] Ostrich 100ms, $n iters"
  $PW optimize_ostrich.py --dt 0.1 --lr 0.3 --num-trials 1 --iterations $n \
      --save results/tim_ost_$n.json > results/tim_ost_$n.log 2>&1
done
for n in 5 15; do
  echo "[$(date +%H:%M:%S)] Semi-Implicit 0.5ms, $n iters"
  $PW optimize_semi_implicit.py --dt 5e-4 --lr 0.1 --num-trials 1 --iterations $n \
      --save results/tim_si_$n.json > results/tim_si_$n.log 2>&1
done
echo "[$(date +%H:%M:%S)] TIMING_ALL_DONE"
