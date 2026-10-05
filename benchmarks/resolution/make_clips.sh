#!/bin/sh
# Build the 10 test clips: 50 frames (every 2nd frame) resized to 512x480,
# from the example videos that ship with the EdgeTAM repo.
cd "$(dirname "$0")"
EDGETAM=${EDGETAM:-../..}
for v in 01_dog 02_hummingbird 03_skateboarder 06_pingpong 08_driving 16_robotarm 19_cyclist 23_racecar 24_clownfish; do
  mkdir -p clips/$v
  ffmpeg -loglevel error -y -i $EDGETAM/examples/$v.mp4 \
    -vf "select='not(mod(n\,2))',scale=512:480:flags=area" -vsync vfr -frames:v 50 -q:v 2 -start_number 0 clips/$v/%05d.jpg
done
mkdir -p clips/bedroom
ffmpeg -loglevel error -y -framerate 30 -start_number 0 -i $EDGETAM/notebooks/videos/bedroom/%05d.jpg \
  -vf "select='not(mod(n\,2))',scale=512:480:flags=area" -vsync vfr -frames:v 50 -q:v 2 -start_number 0 clips/bedroom/%05d.jpg
# 512x512 clip for the speed comparison with upstream (latency_vs_upstream.sh)
mkdir -p clips/bedroom_512
ffmpeg -loglevel error -y -framerate 30 -start_number 0 -i $EDGETAM/notebooks/videos/bedroom/%05d.jpg \
  -vf "scale=512:512:flags=area" -frames:v 60 -q:v 2 -start_number 0 clips/bedroom_512/%05d.jpg
