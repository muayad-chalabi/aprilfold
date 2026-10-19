# Changelog

## 0.1.0 - Unreleased

First release of AprilFold, based on aprilcube 0.3.1.

- Rename the package and command to `aprilfold`.
- Add `--mono`, a foldable build for printers without multi-color support.
  Each face of a cube becomes a black lattice and a white plate printed
  separately. They snap onto a net that prints flat and folds into the cube:
  barbed legs lock each face, and the net latches itself shut with tabs and a
  snap-in lid. Needs the `mono` extra (`manifold3d`). Default tag IDs skip
  markers whose black pixels would print as loose pieces. Larger grids and
  voxel shapes are built as one cube module per voxel, joined by spacers and
  rivets that work for any shape, loops included.
- Add `aprilfold twin`, which prints a kinematic twin of a robot from its URDF
  or xacro. Each link becomes the largest foldable marker cube that fits its
  convex hull, joined to its neighbors by printed revolute (friction knuckle
  with hard stops), continuous, prismatic (bar and sleeve with stops) and fixed
  joints. Writes a detector config per link in that link's frame and a
  `twin.json` with the kinematic chain. Links joined by fixed joints share one
  cube (`--keep-fixed` to separate them), mimic joints follow the one joint the
  twin builds (`--keep-mimic` to build them all), `--scale` resizes the twin,
  `--last-link` stops it at a link (for example before the gripper), and
  `--mesh-dir` finds meshes of a robot file copied without them. Tested on the
  SO-101, ElRobot, KUKA iiwa 14 and Microduck. Needs the `twin` extra.
- Add the foldable example specs `examples/foldable_ring.yaml` and
  `examples/foldable_t_shape.yaml`.
- Remove the `viz` command, whose module was never included.

The entries below are aprilcube's history, kept for reference.

# aprilcube changelog

## 0.3.1 - 2026-10-03

- Add a printable robot calibration cube in `models/calibration_cube`: the
  README's end-effector mount example (1x3x3, `4x4_1000`, 24 mm tags, a 50 mm
  by 10 mm rod to the connector). aiofranka's `aiofranka camera calibrate`
  holds it on a Franka's flange to locate a fixed camera, and the README
  links its 3MF.

## 0.3.0 - 2026-09-08

- Add an optional robot end-effector connector and configurable mounting rod
  on the target's +Z surface, with CLI/YAML settings and 3MF/OBJ exports.
- Support borders in half-cell increments across target generation, detector
  corner geometry, and the voxel designer.
- Expose `MarkerDetection`, `detect_markers()`, and
  `process_marker_detections()` to reuse marker detection across targets that
  share a dictionary. Add `store_latest=False` for custom overlay pipelines.
- Extend the webcam example to discover multiple target configurations and
  draw a combined detection overlay.
- Add T-shaped target size variants, including a dense half-cell-border layout.
- Bundle the connector STL in wheels and source distributions, and keep source
  archives focused on code, documentation, examples, and tests.
- Require OpenCV 4.x to preserve compatibility with its marker detection API.
