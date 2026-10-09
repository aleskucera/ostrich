#!/bin/bash
cd /local/kuceral4/projects/ostrich/experiments/3_gradient_quality_box2 || exit 1
P=/local/kuceral4/projects/ostrich/.venv-mjx/bin/python
CONFIGS=(
  "base|--lr 0.05"
  "lr0.01|--lr 0.01"
  "lr0.02|--lr 0.02"
  "lr0.1|--lr 0.1"
  "lr0.2|--lr 0.2"
  "clipoff|--lr 0.05 --clip-grad-norm 1e9"
  "clip10|--lr 0.05 --clip-grad-norm 10"
  "b2-0.99|--lr 0.05 --beta2 0.99"
  "b2-0.9|--lr 0.05 --beta2 0.9"
  "b1-0.5|--lr 0.05 --beta1 0.5"
)
: > /tmp/hpjobs.txt
i=0
for c in "${CONFIGS[@]}"; do
  tag="${c%%|*}"; args="${c#*|}"
  for t in 20 21; do
    gpu=$((i % 2)); i=$((i+1))
    echo "CUDA_VISIBLE_DEVICES=$gpu $P optimize_mjx.py --dt 0.002 --iterations 30 --num-trials 1 --trial-offset $t $args --save results/hp30_${tag}_t${t}.json > results/hp30_${tag}_t${t}.log 2>&1" >> /tmp/hpjobs.txt
  done
done
echo "queued $(wc -l < /tmp/hpjobs.txt) jobs"
xargs -P 6 -I CMD bash -c CMD < /tmp/hpjobs.txt
echo HPSWEEP_ALL_DONE
