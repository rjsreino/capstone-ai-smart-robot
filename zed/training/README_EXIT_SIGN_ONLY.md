# Landmark Detectors

This version of VICKY uses COCO plus local landmark detectors:

```text
yolov8n.pt
-> COCO safety objects such as person, chair, bottle, backpack

runs/detect/exit_sign_only/weights/best.pt
-> exit_sign only

runs/detect/exit_sign_only_v2/weights/best.pt
-> exit_sign only, trained with the merged exit-sign dataset

runs/detect/door_local_v1/weights/best.pt
-> door_local, mapped to doorway in navigation
```

Door and doorway detection from public datasets was too noisy for this environment, so the server uses the locally captured room-door dataset instead.

## Dataset

The exit-sign dataset lives here:

```text
datasets/exit_sign_only/
  images/
    train/
    val/
  labels/
    train/
    val/
```

All Roboflow directional classes are remapped to:

```text
0 exit_sign
```

The local room-door dataset lives here:

```text
datasets/YOLODataset_door_local/
  images/
    train/
    val/
  labels/
    train/
    val/
```

It uses:

```text
0 door_local
```

## Train

From the repository root:

```powershell
yolo detect train model=yolov8n.pt data=zed/training/exit_sign_only.yaml epochs=30 imgsz=640 name=exit_sign_only
```

Train the local room-door detector:

```powershell
yolo detect train model=yolov8n.pt data=zed/training/door_local.yaml epochs=200 imgsz=640 name=door_local_v2 device=0 degrees=20 perspective=0.0008 scale=0.7 translate=0.15 fliplr=0.5 mosaic=1.0 close_mosaic=20 patience=80 single_cls=True
```

## Run Server

```powershell
python zed/server.py
```

By default, `zed_vision_assistant.py` loads COCO, both exit-sign models, and `door_local_v1`.
