#!/usr/bin/env bash
# Start Isaac Sim (VR experience) with the CloudXR OpenXR runtime and open the scene.
# Run sim/start_cloudxr.sh first. Scene: USD=/path/to/scene.usd (default Hackathon_1stDay.usd).
source ~/.cloudxr/run/cloudxr.env
exec ${ISAAC_ENV:-$HOME/isaac-env-6}/bin/isaacsim isaacsim.exp.base.xr.vr \
    --exec "$(dirname "$0")/open_scene.py" "$@"
