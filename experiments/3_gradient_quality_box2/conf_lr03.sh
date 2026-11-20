#!/bin/bash
cd /local/kuceral4/projects/ostrich/experiments/3_gradient_quality_box2 || exit 1
P=/local/kuceral4/projects/ostrich/.venv-mjx/bin/python
mkjobs() {
  gpu=$1; lo=$2; hi=$3; out=$4
  : > $out
  for t in $(seq $lo $hi); do
    echo "CUDA_VISIBLE_DEVICES=$gpu $P optimize_mjx.py --dt 0.002 --iterations 50 --lr 0.3 --num-trials 1 --trial-offset $t --save results/conf_lr0.3_t${t}.json > results/conf_lr0.3_t${t}.log 2>&1" >> $out
  done
}
mkjobs 0 0 12 /tmp/conf_gpu0.txt
mkjobs 1 13 24 /tmp/conf_gpu1.txt
echo "gpu0 jobs: $(wc -l < /tmp/conf_gpu0.txt)  gpu1 jobs: $(wc -l < /tmp/conf_gpu1.txt)"
xargs -P 3 -I CMD bash -c CMD < /tmp/conf_gpu0.txt &
xargs -P 3 -I CMD bash -c CMD < /tmp/conf_gpu1.txt &
wait
echo CONF_ALL_DONE
