# Datasets

This folder contains the image datasets used to train and test the custom YOLO detection models for the smart robot. The pictures are not general documentation images; they are the training material used to teach the robot to recognize navigation landmarks, especially doors, doorways, and emergency exit signs in indoor environments.

The final goal of these datasets is to improve the robot's camera-based understanding of where it can move and where emergency exits are located. The standard YOLO model is still useful for common objects such as people, chairs, bags, and tables, but doors and exit signs are project-specific enough that we trained our own custom model weights.

## Dataset Layout

Most prepared datasets follow the YOLO object-detection structure:

```text
datasets/
  dataset_name/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
    data.yaml
```

Each image has a matching `.txt` label file when it is part of a YOLO-ready dataset. Label files contain normalized bounding boxes for the target class. Some label files are intentionally empty; these are negative training samples where the image should produce no detection.

## Folder Summary

| Folder | Purpose |
| --- | --- |
| `door_local` | Original local door captures from our target environment. These images were used to test whether a model trained on project-specific doors could recognize the actual doors the robot would encounter. |
| `YOLODataset_door_local` | YOLO-formatted export of the local door dataset. This is a cleaner training-ready version of the local captures. |
| `door_left` | Additional local door images focused on different door positions and viewing angles, including left-side door cases. This helped test whether the detector was overfitting to only one viewpoint. |
| `roboflow_door` | External Roboflow door dataset used to add more variety. It helped expose the detector to door appearances beyond our own limited local captures. |
| `door_combined` | Combined door dataset used for the main door detector experiment. It merges local project images with external door examples and negative samples to produce a more balanced detector. |
| `exit_sign_only` | Exit sign dataset used to train the emergency exit sign detector. It contains positive exit sign examples and negative/background examples. |
| `door_v2` | Reserved or experimental door dataset folder. It is currently empty in this repository snapshot. |

## Current Dataset Counts

| Dataset | Images | Labels | Empty labels |
| --- | ---: | ---: | ---: |
| `door_combined` | 588 | 588 | 33 |
| `door_left` | 591 | 291 | 24 |
| `door_local` | 80 | 40 | 0 |
| `door_v2` | 0 | 0 | 0 |
| `exit_sign_only` | 1901 | 1901 | 186 |
| `roboflow_door` | 297 | 299 | 9 |
| `YOLODataset_door_local` | 40 | 40 | 0 |

For the main combined door dataset, the split is:

| Split | Images | Labels | Empty labels |
| --- | ---: | ---: | ---: |
| `train` | 443 | 443 | 24 |
| `val` | 115 | 115 | 9 |
| `test` | 30 | 30 | 0 |

For the exit sign dataset, the split is:

| Split | Images | Labels | Empty labels |
| --- | ---: | ---: | ---: |
| `train` | 1670 | 1670 | 174 |
| `val` | 231 | 231 | 12 |

## What The Pictures Are Used For

The pictures are used to train custom object detection models. During training, YOLO learns the visual patterns that correspond to the classes we care about:

- `door_local`: doors and passable doorway-like structures relevant to robot navigation.
- exit sign class labels: emergency exit signs used as navigation and safety landmarks.

The robot uses these trained detectors with the camera pipeline so it can identify important indoor features in real time. A door detection can help the system reason about possible passageways, while an exit sign detection can help the system locate emergency direction cues.

## Door Dataset Experiments

We tested several combinations of door images because a door detector trained on only one source can become too narrow. A model trained only on local images may work well in the capture area but fail when lighting, camera angle, frame crop, or door style changes. A model trained only on a broad external dataset may detect generic doors but miss the exact doors and doorway shapes in our deployment environment.

The door experiments were built around these combinations:

- Local-only door captures from `door_local` and `YOLODataset_door_local`.
- Additional left-side and varied-angle local door images from `door_left`.
- External door examples from `roboflow_door`.
- A merged training set in `door_combined`.

The combined approach was the most appropriate for our door detector because it balances two needs:

- It keeps strong examples of our own target doors, which improves performance in the real robot environment.
- It adds external variation, which helps the detector generalize beyond one corridor, one camera angle, or one door appearance.

The resulting `door_combined/data.yaml` defines one class:

```yaml
nc: 1
names:
  0: door_local
```

This class name reflects the project focus: detecting the local door and doorway targets that matter for robot navigation.

## False-Negative And Background Training

The datasets include false-negative/background training through empty YOLO label files. These are images that are included in training but do not contain a target bounding box.

This is important because a detector should learn both when to detect and when not to detect. Indoor environments contain many rectangular objects that can look door-like, such as wall panels, windows, cabinets, posters, hallway edges, or partial frames. Exit sign detection has a similar issue with bright lights, green signs, reflections, labels, and screens.

By including images with empty labels, the model is penalized when it predicts a door or exit sign where there is none. This reduces false positives and makes the detector more reliable during live robot operation.

Examples in this repository:

- `door_combined` contains 33 empty label files.
- `exit_sign_only` contains 186 empty label files.
- `roboflow_door` contains 9 empty label files.
- `door_left` contains 24 empty label files in its YOLO export.

These negative samples make the final detector more conservative and practical. For a robot, this matters because a false door or false exit sign can cause the navigation logic to reason about a landmark that does not actually exist.

## Why We Kept Multiple Dataset Versions

The separate folders document the dataset iteration process. Instead of immediately replacing early datasets, we kept them so we could compare training results across different image combinations:

- Small local datasets showed whether the model could learn our own environment.
- Larger external datasets improved visual diversity.
- Left-side and alternate-angle captures tested viewpoint sensitivity.
- Combined datasets tested whether local relevance and generalization could be balanced.
- Negative samples tested whether the model could avoid incorrect detections.

This dataset history helped us choose a more appropriate training set for the final door and exit sign detection models used by the robot.

## Related Model Outputs

The trained model outputs are stored under `runs/detect/`, including:

- `runs/detect/door_combined_v1/weights/best.pt`
- `runs/detect/exit_sign_only/weights/best.pt`
- `runs/detect/exit_sign_only_v2/weights/best.pt`

These weights are the result of training on the datasets described above and are used by the robot's vision pipeline for custom landmark detection.
