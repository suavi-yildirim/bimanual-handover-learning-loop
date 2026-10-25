#!/usr/bin/env bash
# Side-by-side video (external | left wrist | right wrist, 2x upscaled) for every finished expert episode.
# Usage: sim/make_demo_videos.sh [ROOT=sim/out]   -> <episode>/video.mp4, skipped if it exists
set -u
cd "$(dirname "$0")/.."
ROOT=${1:-sim/out}
find "$ROOT" -name meta.json -path "*ep_*" | while read -r meta; do
    d=$(dirname "$meta")
    [ -f "$d/video.mp4" ] && continue
    label=$(python3 -c "import json;m=json.load(open('$meta'));print(('SUCCESS' if m['success'] else 'FAILED') + '  box offset ' + str([round(v,3) for v in m['box_offset_dx_dy_dz_yaw']]))")
    ffmpeg -v error -y -i "$d/external.mp4" -i "$d/left_wrist.mp4" -i "$d/right_wrist.mp4" \
        -filter_complex "[0][1][2]hstack=3,scale=1344:448:flags=neighbor,drawtext=text='$label':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5" \
        -c:v libx264 -crf 20 -pix_fmt yuv420p "$d/video.mp4" && echo "$d/video.mp4"
done
