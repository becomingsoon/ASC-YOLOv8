# ASC-YOLOv8 — Book Spine Detection for Automated Library Inventory

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

ASC-YOLOv8 is a lightweight one-stage oriented (OBB) detector for **densely
arranged and tilted book spines** in automated library inventory robots. It is
built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
(AGPL-3.0) and adds two modules, implemented as the classes:

- **`MixADown`** (`ultralytics_overlay/nn/modules/block.py`) — the **Mix-ADown**
  adaptive lightweight downsampling module: dual-branch (average pooling +
  mixed max/avg pooling with channel reorganization) replacing a subset of the
  stride-2 `Conv` layers of the backbone and neck.
- **`SCCA`** (`ultralytics_overlay/nn/modules/conv.py`) — the
  **Spatial-Constrained Coordinate Attention** module: coordinate attention
  multiplied by a local depth-wise-convolution spatial mask, embedded at four
  key nodes of the neck.

For compatibility, one-line aliases (`ADown = MixADown`, `GC_CoordAtt = SCCA`)
are kept in the code so that checkpoints trained under the original class names
still load.

## Main results (self-constructed Book Spine Dataset, 1024×1024, GTX 1060)

| Model            | P/%   | R/%   | mAP50/% | mAP50-95/% | Params/M | FLOPs/G | FPS |
|------------------|-------|-------|---------|------------|----------|---------|-----|
| YOLOv8 (baseline)| 99.2  | 98.5  | 99.4    | 86.6       | 3.08     | 21.6    | 43.2|
| + ADown          | 98.9  | 99.2  | 99.4    | 87.8       | 2.66     | 19.4    | 38.6|
| + Mix-ADown      | 98.5  | 98.8  | 99.4    | 88.5       | 2.66     | 19.4    | 37.2|
| + ECA            | 99.1  | 99.1  | 99.4    | 86.6       | 3.08     | 21.6    | 42.1|
| + CBAM           | 98.7  | 99.1  | 99.4    | 87.3       | 3.18     | 21.6    | 40.2|
| + CA             | 99.4  | 99.2  | 99.4    | 88.2       | 3.09     | 21.7    | 40.3|
| + SCCA           | 98.6  | 99.1  | 99.4    | 89.1       | 3.10     | 21.7    | 39.3|
| **ASC-YOLOv8**   | 99.1  | 99.2  | 99.4    | **89.2**   | **2.68** | **19.5**| 35.0|

On the public HRSC2016 benchmark ASC-YOLOv8 reaches **83.7% mAP50-95**,
+1.2% over the baseline (640×640 input).

## Installation

This repository ships the **complete modified Ultralytics package** (v8.3.32,
including the two new modules) so that results are reproducible out of the box.

```bash
git clone <this-repo> && cd <this-repo>
# (optional but recommended) python -m venv .venv && activate it
pip install -e .
python -c "from ultralytics.nn.modules.conv import SCCA; from ultralytics.nn.modules.block import MixADown; print('ok')"
```

Do **not** additionally `pip install ultralytics` - the repo already contains
the (modified) `ultralytics` package and takes precedence in the environment.

## Dataset

- Self-constructed Book Spine Dataset (rotated boxes, 960 train / 240 val):
  <https://doi.org/10.5281/zenodo.21755243>
- Annotate with [roLabelImg](https://github.com/cgvict/roLabelImg); export to
  the Ultralytics OBB format (see `configs/book-spine.yaml`):
  one `*.txt` per image, each line `class x1 y1 x2 y2 x3 y3 x4 y4`
  (normalized polygon of the rotated box).
- HRSC2016 (public): <https://aistudio.baidu.com/datasetdetail/54106>

## Train

```bash
python scripts/train_asc.py configs/book-spine.yaml 500
```

Equivalent CLI:

```bash
yolo task=obb mode=train model=configs/YOLOv8_SCCA3+ADOWN.yaml \
     data=configs/book-spine.yaml epochs=500 imgsz=1024 batch=4 \
     device=0 close_mosaic=10 optimizer=auto pretrained=True
```

## Validate / reproduce the table

```bash
python scripts/evaluate.py runs/asc-yolov8/weights/best.pt configs/book-spine.yaml 1024
```

## Ablation variants

The ablation rows are obtained from the same config:

- **baseline**: remove every `MixADown` row (back to stride-2 `Conv`) and
  every `SCCA` row;
- **+ADown**: use the *stock* `ADown` (max-pool branch only; the original
  pooling line is left commented inside `MixADown.forward`);
- **+Mix-ADown**: this config without the `SCCA` rows;
- **+ECA / +CBAM / +CA / +SCCA**: replace the `SCCA` rows with `ECA`, `CBAM`,
  `CoordAtt` (CA) or keep them (`conv.py` provides all of these classes).

## Notes

- Figures/tables in the paper were produced from the authors' own experimental
  data; plotting scripts are available on request.
- Trained weights (`yolov8+mixadown+scca.pt`, 2,688,694 parameters) can be
  provided on reasonable request / attached to a GitHub Release.
- If you are in an offline environment, launch training from the released
  weights file instead of the config to skip the automatic pretrained download:
  `python scripts/train_asc.py` uses the config; alternatively run
  `yolo task=obb mode=train model=best.pt data=book-spine.yaml epochs=500 imgsz=1024 batch=4`.
- Requires `numpy<2` (Ultralytics 8.3.32 uses the removed `np.trapz` API);
  the pinned dependency in `pyproject.toml` handles this.

## Citation

If you use this code or dataset in your research, please cite:

> N. Wu, W. Zhang, X. Lou, "Book spine detection for automated library
> inventory using ASC-YOLOv8", *Scientific Reports*, 2026 (under review).

## License

This project is based on Ultralytics YOLOv8 (AGPL-3.0) and is distributed under
the [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) license.

## Acknowledgements

This work was supported by the Library Society of Zhejiang Province Research
Project under grant no. Ztx2024C-11.
