#!/usr/bin/env bash
# Reference pools with margins for the margin estimator: d in {10, 20},
# push a in {0.6, 0.7, 0.8}, 3000 antithetic pairs each, on dasenka.
# Detached with setsid + nohup: survives closing the laptop / ssh.
#   ./launch_margin_ref.sh          sync code and launch
#   ./launch_margin_ref.sh fetch    copy finished results back
set -euo pipefail
SSH="ssh -o RemoteCommand=none -o RequestTTY=no -o ForwardX11=no"
DIR=/local/kuceral4/projects/exp12-smoothed
PY=/local/kuceral4/projects/ostrich/.venv/bin/python
cd "$(dirname "$0")"
if [[ "${1:-}" == fetch ]]; then
  rsync -av -e "$SSH" "dasenka:$DIR/results/night/margin_ref_*" results/night/
  exit
fi
rsync -av -e "$SSH" ./*.py "dasenka:$DIR/"
$SSH dasenka "cd $DIR && mkdir -p results/night && for d in 10 20; do for a in 0.6 0.7 0.8; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONPATH=. setsid nohup $PY margin_probe.py \$d \$a 3000 results/night/margin_ref_d\${d}_a\$a.npz \
    > results/night/margin_ref_d\${d}_a\$a.log 2>&1 < /dev/null &
done; done; sleep 2; pgrep -af margin_probe"
