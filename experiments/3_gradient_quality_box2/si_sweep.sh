#!/bin/bash
cd /local/kuceral4/projects/ostrich/experiments/3_gradient_quality_box2 || exit 1
P=/local/kuceral4/projects/ostrich/.venv/bin/python
: > /tmp/si_gpu0.txt; : > /tmp/si_gpu1.txt
i=0
for lr in 0.02 0.1 0.3; do
  for t in 0 1 2 3 4 5 6 7; do
    gpu=$((i % 2)); i=$((i+1))
    echo "CUDA_VISIBLE_DEVICES=$gpu $P optimize_semi_implicit.py --dt 5e-4 --iterations 30 --lr $lr --num-trials 1 --trial-offset $t --save results/si30_lr${lr}_t${t}.json > results/si30_lr${lr}_t${t}.log 2>&1" >> /tmp/si_gpu${gpu}.txt
  done
done
echo "gpu0 jobs: $(wc -l < /tmp/si_gpu0.txt)  gpu1 jobs: $(wc -l < /tmp/si_gpu1.txt)"
xargs -P 3 -I CMD bash -c CMD < /tmp/si_gpu0.txt &
xargs -P 3 -I CMD bash -c CMD < /tmp/si_gpu1.txt &
wait
echo SI_SWEEP_ALL_DONE
