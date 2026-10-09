#!/bin/bash
cd /local/kuceral4/projects/ostrich/experiments/3_gradient_quality_box2 || exit 1
P=/local/kuceral4/projects/ostrich/.venv-mjx/bin/python
CONFIGS=(
  "lr0.3|--lr 0.3"
  "lr0.5|--lr 0.5"
  "lr0.8|--lr 0.8"
  "lr0.3-clipoff|--lr 0.3 --clip-grad-norm 1e9"
  "lr0.5-clipoff|--lr 0.5 --clip-grad-norm 1e9"
  "lr0.3-clip10|--lr 0.3 --clip-grad-norm 10"
  "lr0.3-b1-0.5|--lr 0.3 --beta1 0.5"
)
: > /tmp/hpjobs2.txt
i=0
for c in "${CONFIGS[@]}"; do
  tag="${c%%|*}"; args="${c#*|}"
  for t in 18 20 21; do
    gpu=$((i % 2)); i=$((i+1))
    echo "CUDA_VISIBLE_DEVICES=$gpu $P optimize_mjx.py --dt 0.002 --iterations 30 --num-trials 1 --trial-offset $t $args --save results/hp30_${tag}_t${t}.json > results/hp30_${tag}_t${t}.log 2>&1" >> /tmp/hpjobs2.txt
  done
done
echo "CUDA_VISIBLE_DEVICES=1 $P optimize_mjx.py --dt 0.002 --iterations 30 --num-trials 1 --trial-offset 18 --lr 0.05 --save results/hp30_base_t18.json > results/hp30_base_t18.log 2>&1" >> /tmp/hpjobs2.txt
echo "queued $(wc -l < /tmp/hpjobs2.txt) jobs"
xargs -P 6 -I CMD bash -c CMD < /tmp/hpjobs2.txt
echo HPSWEEP2_ALL_DONE
