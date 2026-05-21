#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$DIR/ball_throw/run_all.sh"
bash "$DIR/sliding_box/run_all.sh"
bash "$DIR/helhest/run_all.sh"
