# HTCV, Mouse and Lockbox in One Metric 3D Frame

Seminar project (TU Berlin / Science of Intelligence, *Topic 16, Real Mouse Data*) reconstructing a mouse and the lockbox mechanisms it manipulates into a single, metric three-dimensional coordinate system, then reading the solving stage of the box out of that frame.

Mice solving mechanical lockboxes are usually analysed with 2D keypoints, which cannot say how far a mechanism has travelled or whether the animal was actually at it when it moved. A pixel distance between a paw and a lever is not a physical distance, and when a paw disappears behind the box in one camera, a single-view analysis cannot recruit the cameras that still see it.

This pipeline addresses that by calibrating against the apparatus itself. The lockbox is rigid and its CAD geometry is public, so clicking a handful of known corners recovers the camera poses and fixes the world scale in millimetres, with no checkerboard and no calibration rig. That matters because the dataset ships no calibration at all.

**Main finding.** The mechanism state does not need to be learned. Projecting each tracked mechanism keypoint onto its own kinematic axis gives the state directly, in millimetres or radians, and it beats every learned decoder we trained on mouse pose.

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Kenny-Dean/mouse-lockbox-3d.git
cd mouse-lockbox-3d
uv sync                                                   # creates .venv from uv.lock
uv run jupyter lab                                        # everything runs in notebooks
```

## Data

None of the data is in this repository (`/data` is git-ignored). You need to fetch it yourself.

### 1. The video dataset (required)

**Mouse Lockbox Dataset**, Reiske et al. 2025, [arXiv:2505.15408](https://arxiv.org/abs/2505.15408).
Download it from TU Berlin DepositOnce at <https://doi.org/10.14279/depositonce-23850>

This project uses the LB#2 combined lockbox (lever, sliding stick, ball, sliding door), mouse
291, two sessions on consecutive days. Place them as follows, keeping the dataset's original
filenames. (mlb-labeled-mouse291.tar (3.98 GB))

```
data/validation/scene1/     # 18 June 2021, 8190 frames, 273 s, pipeline built here
  2021-06-18_07-28-45_segment1_mouse291_combined_top-down-view.avi
  2021-06-18_07-28-45_segment1_mouse291_combined_side-view.avi
  2021-06-18_07-28-45_segment1_mouse291_combined_front-view.avi
  2021-06-18_07-28-45_segment1_mouse291_combined_labels.csv    # ground-truth states

data/validation/scene2/     # 17 June 2021, 5670 frames, 189 s, held out, never trained on
  2021-06-17_07-18-24_segment1_mouse291_combined_top-down-view.avi
  ...  (same four files)
```

### 2. The lockbox CAD geometry (optional)

`MouseLockBox/` is a vendored copy of the hardware deposit
([RefinementReferenceCenter/MouseLockBox](https://github.com/RefinementReferenceCenter/MouseLockBox)), holding 3D-printable STL files, the construction manual and demo GIFs/videos.

You only need it if you calibrate a different lockbox variant. The calibration points for LB#2 combined are already hard-coded in `01_lockbox_calibration` (measured off the STLs in Blender/MeshLab) and written to `points_3d.json`. For a new variant, read fresh corner coordinates off the STL, in millimetres relative to a chosen origin.

## Running the pipeline

Notebooks live in `notebooks/` and run in numeric order. Each consumes pickle/JSON artifacts written by earlier stages and writes its own under `data/<stage>/<scene>/`. Paths are relative, so the notebooks run as-is from `notebooks/`. The scene name, however, is a hard-coded literal in most of them, so running a different session means editing those strings.

| Notebook | Role | Key output under `data/…/<scene>/` |
|---|---|---|
| `00_multiview_3d_pipeline` | Preprocess (CLAHE on IR top-down), run SuperAnimal 2D inference on every view, pick the best top view | `multiview_output/`, `pipeline_export/2d_state.pkl` |
| `01_lockbox_calibration` | Click CAD points -> camera poses via PnP -> metric world frame | `lockbox_calibration/lockbox_calibration.toml` |
| `02_triangulate_and_render` | Triangulate 2D -> 3D mouse pose with the aniposelib `CameraGroup` | `triangulate_render/pipeline_state.pkl` |
| `03_nn_pose_inpainting` | Torch NN filling missing/occluded 3D keypoints (evaluated, not adopted) | updated 3D pose |
| `04_lockbox_state_labeling_prep` | Define the mechanism **state schema** (keypoints + 1-DOF axis), build and label per-view DLC projects | `lockbox_dlc/state_schema.json`, `selected_frames.json` |
| `05_build_perview_projects` | Prune, train and validate the per-view DLC models (`top`/`side`/`front`) | trained models under `lockbox_dlc/…/perview/` |
| `06_lockbox_state_extraction` | Run per-view DLC, triangulate mechanism keypoints -> scalar state per mechanism -> solving stage | `lockbox_state/` |
| `07_predict_lockbox_states` | Train the decoders (XGBoost, 1D-CNN, transformer) on extracted states | `lockbox_state_prediction/` |
| `08_hand_mechanism_interaction` | Paw/nose vs. mechanism state-change coincidence (descriptive, predicts nothing) | `hand_mechanism_interaction/` |
| `09_validate_unseen_video` | Re-run the whole pipeline on the held-out session | `validation/` |

### Two stages need a human

- **`01_lockbox_calibration`** requires you to *click* six corresponding lockbox points in each of   the three views. There is no automatic target detection.
- **`04`/`05`** require hand-labelling mechanism keypoints in DeepLabCut, 665 frames per view   (≈ 8 % of scene 1) and about 2000 placed labels across the three views.

## Credits

- **Dataset**. Reiske, Boon, Andresen, Traverso, Hohlbaum, Lewejohann, Thöne-Reineke, Hellwich, Sprekeler. *Mouse Lockbox Dataset, Behavior Recognition for Mice Solving Lockboxes*, 2025. [arXiv:2505.15408](https://arxiv.org/abs/2505.15408) [DOI](https://doi.org/10.14279/depositonce-23850)
- **Lockbox hardware**. Hohlbaum, Andresen, Boon, Kahnau, Mieske, Sprekeler, Hellwich, Thöne-Reineke, Lewejohann, [RefinementReferenceCenter/MouseLockBox](https://github.com/RefinementReferenceCenter/MouseLockBox)
- **2D pose**. [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) and SuperAnimal (Mathis et al. 2018, Ye et al. 2024)
- **Triangulation**. [aniposelib](https://github.com/lambdaloop/aniposelib) (Karashchuk et al. 2021)
