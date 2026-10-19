#!/usr/bin/env bash
# Optimization on the narrow-window box (opt_highd boxf) on dasenka:
# d in {10, 20}, methods M2 / linreg / gated / margin, N = d pairs, lr scale
# in {0.5, 1.0}, seeds 0..9 split over two processes. Detached (setsid nohup).
#   ./launch_boxf_opt.sh          sync code and launch
#   ./launch_boxf_opt.sh fetch    copy finished results back
set -euo pipefail
SSH="ssh -o RemoteCommand=none -o RequestTTY=no -o ForwardX11=no"
DIR=/local/kuceral4/projects/exp12-smoothed
PY=/local/kuceral4/projects/ostrich/.venv/bin/python
cd "$(dirname "$0")"
if [[ "${1:-}" == fetch ]]; then
  rsync -av -e "$SSH" "dasenka:$DIR/results/opt_highd/boxf*" "dasenka:$DIR/results/opt_highd/logs_boxf" results/opt_highd/
  exit
fi
rsync -av -e "$SSH" ./*.py "dasenka:$DIR/"
$SSH dasenka "cd $DIR && mkdir -p results/opt_highd/logs_boxf && for d in 10 20; do for m in zeroth_order linreg gated margin; do for c in 0.5 1.0; do for s in 0 5; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. setsid nohup $PY opt_highd.py boxf \$d 0 \$m \$d \$c \$s \$((s + 5)) boxf_d\$d \
    > results/opt_highd/logs_boxf/d\${d}_\${m}_lr\${c}_s\$s.log 2>&1 < /dev/null &
done; done; done; done; sleep 2; pgrep -fc 'opt_highd.py boxf'"
