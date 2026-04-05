# Blender workflow (BVH from this API)

## Defaults (recommended)

- **`bvh_root_local=true`** (default on all `POST` routes): frame‑0 Hips is shifted to the origin for the whole skeleton. Reduces MediaPipe world drift and makes retargeting easier. Use `bvh_root_local=false` only if you need absolute world translation.

- **ZIP export** (`/v1/bvh_preview.zip`): by default includes **`{name}_joints_world_cm.csv`** — the same smoothed 22‑joint positions (centimeters) that feed `positions_to_bvh`. Plot or script against this to debug; motion is **not** identical to BVH Euler playback (FK adds twist ambiguity).

## Import

1. **File → Import → Motion Capture (.bvh)** (or drag the `.bvh` into the 3D View).
2. **Axis / up**: MediaPipe world is **Y‑up**; Blender is **Z‑up**. If the armature lies on its side, select the imported armature, **Object mode**, rotate (often **−90° on X** on the object) and **Apply** rotation if needed.
3. **Scale**: BVH positions and `OFFSET` values are in **centimeters** (meters × 100 from the pipeline). If the rig is tiny or huge, check **Scene Properties → Units** or scale the armature object (e.g. **0.01** if you treat 1 Blender unit = 1 m).

## Evaluation order

1. Import BVH and play the action on the **default stick/skeleton** from the file.
2. If that looks wrong, compare mentally (or in a spreadsheet) with **`_joints_world_cm.csv`** from the same export.
3. Only then **retarget** to a humanoid (Rigify, Auto‑Rig Pro, etc.); retargeting adds its own approximations.

## GIF vs BVH

The preview GIF draws **image‑space** joints (with matching topology and smoothing on the image track). The BVH encodes **world‑space** joints as **hierarchical rotations**. They are related but not pixel‑or‑rotation identical; for a ground‑truth view of the **BVH file**, use Blender’s viewport or render, not the GIF alone.

## Why the HIERARCHY block looks “unchanged”

`bvh_root_local` applies a **rigid translation** to the whole skeleton (subtract frame‑0 Hips from **every** joint). That **does not change** any bone vector `child − parent`, so **all `OFFSET` lines in `HIERARCHY` stay the same** as before. Only **`MOTION`** changes: with root‑local, the root’s **Xposition Yposition Zposition** start near **0,0,0** and stay small (in‑place motion); without it, those three numbers are large and drift in MediaPipe world space.

**GIF vs Blender with root‑local:** The GIF still shows the **person moving in the camera frame**. The BVH moves the **rig near the origin** on purpose. So the clip and the armature will **not** line up spatially even when the export is correct.

## How to verify an export

1. Open **`_joints_world_cm.csv`**: row **0** should have **`Hips_x_cm,Hips_y_cm,Hips_z_cm` = 0,0,0** when root‑local is on (default).
2. Open **`test_dance.bvh`** `MOTION` section: the **first three numbers** on each line are root translation (cm). Frame **0** should be **0 0 0** (or tiny float noise) with defaults.
3. HTTP responses include header **`X-Bvh-Root-Local: true|false`** so you can confirm what the server used.
