import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=["person", "chair", "bottle"],
    max_samples=500
)

dataset.export(
    export_dir="dataset_yolo",
    dataset_type=fo.types.YOLOv5Dataset
)

print("Export done")