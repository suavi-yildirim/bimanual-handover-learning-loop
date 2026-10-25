#!/usr/bin/env bash
# Start the CloudXR OpenXR runtime for the Quest 3 (keep this terminal open).
# Then, in a second terminal: source ~/.cloudxr/run/cloudxr.env and start Isaac Sim.
ISAAC_ENV=${ISAAC_ENV:-$HOME/isaac-env-6}
PREBUNDLE=$ISAAC_ENV/lib/python3.12/site-packages/isaacsim/exts/isaacsim.replicator.teleop/pip_prebundle
export PYTHONPATH=$HOME/.cloudxr-pydeps:$PREBUNDLE
exec $ISAAC_ENV/bin/python -m isaacteleop.cloudxr "$@"
