# Source and license

The downloaded upstream is Ultralytics YOLOv5 v7.0, commit
915bbf294bb74c859f0b41f1c23bc395014ea679, from
https://github.com/ultralytics/yolov5, licensed under GPL-3.0 at this commit.
Keep its LICENSE file in every prepared checkout. upstream.patch modifies those
GPL-3.0 source files and is distributed with the same upstream license.
The official weights come only from the fixed v7.0 release URL recorded in
upstream.lock.json. No weights or upstream checkout are committed here.

The public evaluator is inherited unchanged from this project's comparison base;
its separate Ultralytics source/license provenance remains in
../evaluation/native_provenance.json and ../evaluation/native_metrics.py.
