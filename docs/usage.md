# AprilFold User Guide

AprilFold makes fiducial targets you can print on any single-color printer:
every cube is a flat net that folds shut, with marker faces that snap on. It
also turns a robot's URDF into a printed twin built from those cubes, for
teleoperating the robot from a camera alone.

AprilFold is built on [aprilcube](https://github.com/younghyopark/aprilcube) by
Younghyo Park (MIT License). Marker layout, voxel targets, the browser designer
and the detector come from aprilcube; see [Credits](#credits-and-citation).

- [Install](#install)
- [Foldable cubes](#foldable-cubes)
- [Foldable voxel targets](#foldable-voxel-targets)
- [Robot twins](#robot-twins)
- [Tracking](#tracking)
- [Reference](#reference)

## Install

```bash
pip install 'aprilfold[mono]'     # foldable cubes and voxel targets
pip install 'aprilfold[twin]'     # robot twins (adds trimesh and scipy)
```

From a checkout: `pip install -e '.[twin,test]'`. The geometry is built with
[manifold3d](https://github.com/elalish/manifold); detection needs only OpenCV
and NumPy.

## Foldable cubes

```bash
aprilfold generate --grid 1x1x1 --dict 4x4_50 --tag-size 30 --mono -o my_cube
```

`--mono` selects the foldable, single-color build. Without it, `generate`
writes aprilcube's multi-color 3MF, which needs a printer that changes
filament.

### What you print

Each face is three parts, and the six net panels are one print:

| File | Print in | Parts |
|------|----------|-------|
| `print_net.3mf` | any color | Six panels joined by living hinges, printed flat |
| `print_white.3mf` | white | One plate per face; the white pixels stand up and the lattice drops between them |
| `print_black.3mf` | black | One lattice per face: the marker's black pixels as one plate, with two barbed legs |

`assembly_preview.3mf` and `mono_preview.png` show the assembled cube. They are
for checking only.

### Assembly

1. Fold the two `X` panels up.
2. Fold the `Y` panels: their tabs slide into slots inside the `X` panels.
3. Close the `+Z` lid until both of its tabs click past the flexure beams in
   the `Y` panels. A tab can only enter a panel that is already folded, so the
   order matters; once the lid clicks, every panel is trapped.
4. Lay each white plate on its panel (the panel is engraved with the tag ID).
5. Press each black lattice into its plate. Its legs pass through plate and
   panel, push aside a flexure beam and hook behind it, so one click clamps
   all three layers.

The net holds itself without glue or tape. All parts are mitered at 45
degrees, so no seams show across a face.

### Marker IDs and border size

White data cells sit inside the black border ring, so the black part is the
one dropped in, and its black cells must connect edge to edge. When `--ids` is
omitted, AprilFold skips markers whose black pixels split apart (in `4x4_50`,
IDs 1-6 instead of 0-5); passing such an ID is an error.

The lattice legs hang under the border ring and must fit inside the folded
net, so small markers need a wider border. The generator reports the smallest
`--border-cell` that fits, for example 1.5 cells for a 24 mm `4x4_50` tag or a
30 mm `apriltag_36h11` tag.

### Fit tuning

Adjust clearances for your printer in a YAML spec (`fit_clearance_mm` is also
`--mono-clearance`):

```yaml
mono:
  fit_clearance_mm: 0.15   # lattice-to-plate gap per side
  slot_clearance_mm: 0.2   # around legs and tabs
  inlay_mm: 1.6            # lattice thickness
  base_mm: 1.2             # white floor under the lattice
  panel_mm: 3.2            # net panel thickness
  hinge_mm: 0.4            # living-hinge thickness
```

The detector `config.json` is the same as for a multi-color cube, so
[Tracking](#tracking) works the same for both.

## Foldable voxel targets

Grids other than 1x1x1 and voxel shapes (`voxel_grid`, `voxel_cuboids`) are
built from cube modules, one per voxel. Each module is the foldable cube above,
with markers on the faces that show. Voxels that show no marker, such as the
center of a 3x3x3, are left out. This works for any shape, including ones with
no single net and closed loops.

```bash
aprilfold generate --grid 3x3x3 --dict apriltag_36h11 --tag-size 30 --border-cell 1.5 --mono
aprilfold generate examples/foldable_ring.yaml        # 3x3 frame, center missing
aprilfold generate examples/foldable_t_shape.yaml     # five-cube T
```

Print `print_black.3mf` in black and `print_white.3mf` in white as before;
`print_nets.3mf` (one net per module) and `print_joints.3mf` (spacers and
rivets) can be any color.

Neighboring modules share a face. A printed spacer fills the gap between their
panels, and a rivet with two barbed legs joins them: it is pushed from inside
one module, through its panel and the spacer, into the neighbor's panel, using
the same leg slots as the black lattice. Every module's lid opens on a face
that shows a marker, and each joint is riveted from the module placed later,
so the last module slides in with flat faces and its rivets go in afterwards.
The output README lists the assembly order.

Because each voxel carries its own border, a `--grid` target built this way
puts markers `2 x --border-cell` cells apart rather than `--margin-cell` apart.
Its `config.json` lists every marker's corners explicitly, so detection is
unchanged.

## Robot twins

`aprilfold twin` turns a robot description into a printed kinematic twin.
Every link becomes a foldable marker cube, the largest that fits inside the
link's convex hull. Neighboring cubes are joined by printed joints at the URDF
joint positions. Move the twin by hand while a camera tracks the cubes, and you
have every link's pose, and so the joint angles, to drive the real robot.

```bash
pip install 'aprilfold[twin]'
aprilfold twin so101.urdf.xacro -o so101_twin
```

### Input

The input is a URDF or a xacro file (expanded with the `xacro` command, so
source your ROS environment first). Mesh paths may be `package://`, `file://`
or relative. When a package is not found next to the robot file or on
`AMENT_PREFIX_PATH`, pass `--package-path name=/path/to/name`. For a robot file
copied away from its meshes, `--mesh-dir DIR` searches DIR for each mesh by its
trailing path. Links whose visual meshes cannot be loaded (for example `.dae`
without `pycollada`) use their collision geometry.

Before planning, the robot is reduced to what a twin needs:

- **Rigid groups.** Links joined by `fixed` joints move as one body and get
  one cube (a servo bolted to its bracket, a gripper flange, a ground plate).
  Pass `--keep-fixed` to give each its own cube, joined by a printed strut.
  Links with no geometry (such as `world`) are always folded away.
- **Mimic joints.** Joints that `<mimic>` another move with it, so the twin
  builds one joint of each group, the one whose link is largest, and leaves out
  the others with their links. `twin.json` lists them under `mimic_joints` as
  `value = multiplier * kept joint + offset`. Pass `--keep-mimic` to build them
  as independent joints.
- **`--last-link LINK`** ends the twin at that link and leaves out everything
  beyond it, for example `--last-link wrist` for an SO-101 without its gripper.

### Joint hardware

| Joint | Printed hardware |
|-------|------------------|
| `revolute` | Two knuckle discs on a printed bolt with a wing nut; tighten the nut until the joint holds its pose. A pin on the child disc runs in an arc groove in the parent disc that ends at the URDF limits. The bolt's flat keys it to the nut's disc, so moving the joint never loosens the nut. |
| `continuous` | The same knuckle with a full-circle groove (no stops). |
| `prismatic` | A square bar on the parent and a sleeve on the child. The bar's fixed end and a pinned cap at its free end stop the sleeve at the URDF limits. |
| `fixed` | With `--keep-fixed`: one rigid strut that snaps into both cubes. |

The hardware sits on the URDF joint axis, and the cubes are kept clear of it.
For each joint the generator searches the knuckle's position along the axis,
which disc is on top, the bolt direction, and the strut shapes (straight, bent,
around the side of a cube, or swung around the axis to pass a neighbor). It
keeps the layout with the least material where nothing collides anywhere in the
joint range. It also keeps clear of the slot where the next joint's knuckle
will go, and of the links further up the chain as each of their joints moves;
when that leaves no layout, it settles for being clear of the parent's own
joint. Afterwards it poses the whole twin through its joint ranges and warns
about any pose where two links touch but the robot's own meshes do not.

### Size and scale

A foldable cube needs an edge of about 35 mm with 4x4 markers. Links whose
hull is smaller still get a cube of that size, which then sticks out of the
link; the generator says by how much. Such a cube is placed to overhang its
link evenly and to stay clear of the knuckles and of the other cubes.
`--scale` enlarges the whole twin (joint angles are unchanged), which gives
small links room for their cube and every joint room for its knuckle. If a
joint finds no layout, the cubes are planned again with that joint's room kept
strictly clear before the generator gives up and names what blocked it.

How small a robot can get depends on how its joint spacing compares with the
35 mm cubes. Tested robots:

| Robot | Joints | Builds at | Notes |
|-------|--------|-----------|-------|
| SO-101 arm | 6 revolute | 1:1 | wrist, gripper and jaw cubes stick out of their links |
| ElRobot arm | 7 revolute, gripper slider | 1:1 | servos merge into their brackets; the gripper gear and second jaw follow the first jaw (mimic) |
| KUKA iiwa 14 | 7 revolute | 0.3 and up | at 0.2 its joints are about 40 mm apart and no layout is found |
| Microduck biped | 15 revolute | 2 | at 1:1 the 16 cubes are larger than most of its links and collide |

[models/so101_twin](../models/so101_twin/README.md) is a complete 1:1 SO-101
twin.

### Output

| File | Contents |
|------|----------|
| `twin.json` | Kinematic chain (joint origins in mm, axes, limits), each cube's pose in its link frame, hardware, merged links, mimic joints and warnings |
| `links/<link>/config.json` | Detector config for one link; marker corners are in that link's frame, so a detected pose is the link's pose |
| `print_black.3mf`, `print_white.3mf`, `print_nets.3mf` | Marker parts and one folding net per link |
| `print_connectors.3mf` | Mount plates with struts and knuckles or sliders (print with supports) |
| `print_hardware.3mf` | Bolts, wing nuts, slider caps and pins |
| `assembly_preview.3mf`, `twin_preview.png` | The assembled twin, for checking |
| `README.md` | Links, joints, print list, assembly steps and warnings for this twin |

A face that carries a joint mount gets a connector instead of a marker: the
connector's legs click into the panel the same way a black lattice does.

### Reading joint angles

Run one detector per link (`links/<link>/config.json`). Each gives the link's
pose in the camera frame, `T_cam_link`. For a joint between `parent` and
`child`, the motion is

```text
M = inv(T_cam_parent @ origin) @ T_cam_child
```

with `origin` from `twin.json`. For a revolute joint the angle is the rotation
of `M` about the joint `axis`; for a prismatic joint it is `M`'s translation
along `axis` (mm). Clamp to the joint's limits and send the values to the robot.

## Tracking

```python
import aprilfold

det = aprilfold.detector(
    "my_cube/config.json",
    {"fx": 800, "fy": 800, "cx": 320, "cy": 240},
)
result = det.process_frame(frame)          # BGR numpy array

if result["success"]:
    T = result["T"]                        # 4x4 camera-frame pose, mm
    error = result["reproj_error"]         # reprojection error in pixels
    faces = result["visible_faces"]
```

### `aprilfold.detector(cube_cfg, intrinsic_cfg, **kwargs)`

Creates a `CubePoseEstimator`. Configs may describe cuboids, voxel targets or
twin links.

| Arg | Type | Description |
|-----|------|-------------|
| `cube_cfg` | `str \| Path` | Path to `config.json` or a generated model directory |
| `intrinsic_cfg` | `str \| Path \| dict \| np.ndarray` | Camera intrinsics |
| `extrinsic` | `np.ndarray \| None` | 4x4 world-to-camera transform (default: `None`) |
| `enable_filter` | `bool` | Kalman temporal smoothing (default: `True`) |
| `filter_config` | `KalmanFilterConfig \| None` | Custom filter tuning |
| `dist_coeffs` | `np.ndarray \| None` | Override distortion coefficients |
| `fast` | `bool` | Faster detection for real-time use (default: `False`) |

Intrinsics can be a calibration JSON path (keys `camera_matrix`, optional
`dist_coeffs`), a dict with `fx`, `fy`, `cx`, `cy`, or a 3x3 camera matrix.

### World-frame poses

```python
det = aprilfold.detector("my_cube", intrinsics, extrinsic=T_world_cam)
T_world_obj = det.world_pose(det.process_frame(frame))   # 4x4 or None
```

### Async detection

```python
det = aprilfold.detector("models/my_target", intrinsics, fast=True)
det.start_async()
while True:
    ok, frame = cap.read()
    det.submit_frame(frame)        # non-blocking; replaces any pending frame
    result = det.get_latest()      # last completed result, instantly
det.stop_async()
```

If detection is slower than capture, intermediate frames are skipped so the
detector always works on the freshest frame.

### 3D view with viser

```bash
pip install viser trimesh
```

```python
server = det.build_viser(port=8080)   # renders the latest result in the browser
```

The scene shows the world, camera and object frames, a ground grid, and the
target mesh from `mujoco/cube.obj`. See `examples/webcam_simple.py` and
`examples/webcam_viser.py`.

### Direct class access

```python
from aprilfold import CubePoseEstimator, CubeConfig, KalmanFilterConfig
from aprilfold.generate import TagPatternGenerator, CubeMeshBuilder, ThreeMFWriter
```

## Reference

### `generate` options

| Arg | Default | Description |
|-----|---------|-------------|
| `spec` / `-c, --config` | - | YAML generation spec |
| `-g, --grid` | `1x1x1` | Tags per dimension: `WxHxD` |
| `-d, --dict` | `4x4_50` | ArUco/AprilTag dictionary |
| `-t, --ids` | auto | Tag IDs: range (`0-23`) or comma-separated |
| `--tag-size` | `30` | Tag size in mm |
| `--cell-size` | - | Cell size in mm, alternative to `--tag-size` |
| `--margin-cell` | `1` | Gap between adjacent tags, in cells |
| `--border-cell` | `1` | Outer border per face edge, in 0.5-cell steps |
| `-o, --output` | `aruco_cube` | Output directory |
| `--mono` | - | Foldable single-color build |
| `--mono-clearance` | `0.15` | Per-side gap in mm between the black lattice and the white plate |
| `--extruder` | `1` | Bambu Studio extruder number (multi-color build) |
| `--invert` | - | Swap black and white |
| `--end-effector-connector` | - | Attach `assets/end_effector_connector.stl` on the +Z surface |
| `--connector-rod-length` | - | Rod length in mm from the +Z surface to the connector |
| `--connector-rod-radius` | `10` | Connector rod radius in mm |
| `--connector-stl` | `assets/end_effector_connector.stl` | Alternate connector STL |

Half-cell borders keep markers centered by rendering each logical marker cell
as a 2x2 surface raster: `--border-cell 0.5` leaves half a logical cell at each
outer edge while keeping the physical tag size.

### `twin` options

| Arg | Default | Description |
|-----|---------|-------------|
| `robot` | - | URDF or xacro file |
| `-o, --output` | `<robot>_twin` | Output directory |
| `--scale` | `1` | Uniform scale of the twin |
| `-d, --dict` | `4x4_250` | Marker dictionary |
| `-t, --ids` | auto | Tag IDs |
| `--package-path PKG=DIR` | - | Where to find a ROS package (repeatable) |
| `--mesh-dir DIR` | - | Where to search for meshes missing next to the robot file (repeatable) |
| `--last-link LINK` | - | Leave out every link beyond this one |
| `--keep-fixed` | - | Separate cubes for links joined by fixed joints |
| `--keep-mimic` | - | Build mimic joints as independent joints |
| `--mono-clearance` | `0.15` | Lattice-to-plate gap per side |
| `--quick` | - | Fewer cube orientations and collision poses (faster, for trying things out) |

### YAML specs

`aprilfold generate` reads YAML specs. Shapes are `cuboid`, `voxel_cuboids`
and `voxel_grid`; add `mono: true` (or a `mono:` block) for the foldable build.

```yaml
output: models/t_shape_target
shape:
  type: voxel_cuboids
  voxel_size_mm: 24
  cuboids:
    - name: stem
      origin: [1, 0, 0]
      size: [1, 1, 3]
    - name: crossbar
      origin: [0, 0, 2]
      size: [3, 1, 1]
dictionary: 4x4_50
markers:
  ids: 0-63
size:
  tag_size_mm: 18
layout:
  margin_cells: 1
  border_cells: 1
mono: true
```

`examples/foldable_ring.yaml` and `examples/foldable_t_shape.yaml` are
foldable specs. The other shapes in `examples/` (`t_shape`, `l_shape`,
`stair_step`, `u_shape`, `plus_shape`, `spiral_tower`, `zigzag_snake`,
`window_frame`, `chair`) come from aprilcube and fix their tag IDs for the
multi-color build; for a foldable build, drop their `markers: ids:` line so
AprilFold can pick IDs whose black pixels stay in one piece, and add
`mono: true`. An `attachment:` block (`type:
end_effector_connector`, `rod_length_mm`, `rod_radius_mm`) adds a robot mount.

### Voxel designer

```bash
aprilfold web            # open the browser designer
aprilfold web --no-open  # print the local HTML path
```

Edit voxel occupancy, preview marker placement, and download a YAML spec for
`aprilfold generate`.

### Grids and dictionaries

| Grid | Shape | Tags |
|------|-------|------|
| `1x1x1` | Cube | 1 per face, 6 total |
| `2x2x2` | Cube | 4 per face, 24 total |
| `5x4x1` | Flat box | 20 on top and bottom, narrow side strips |
| `1x1x3` | Pillar | 3 on the tall sides, 1 on the caps |

**ArUco:** `4x4_50`, `4x4_100`, `4x4_250`, `4x4_1000`, `5x5_*`, `6x6_*`,
`7x7_*`, `aruco_original`. **AprilTag:** `apriltag_16h5`, `apriltag_25h9`,
`apriltag_36h10`, `apriltag_36h11`.

### Output files

```text
my_cube/
  print_net.3mf         # foldable build: net, white plates, black lattices
  print_white.3mf
  print_black.3mf
  assembly_preview.3mf  # for checking, not for printing
  mono_preview.png
  config.json           # what the detector needs
  thumbnail.png         # dimensions, tag IDs and axes
  README.md
  mujoco/               # cube.xml (MJCF), cube.obj/.mtl, cube_atlas.png
```

The multi-color build writes a single `cube.3mf` instead of the three print
files. The MJCF and OBJ use the detector's frame: origin at the target's
bounding-box center, in meters (`python -m mujoco.viewer --mjcf
my_cube/mujoco/cube.xml`).

A cuboid `config.json` maps faces to tag IDs (`faces`, `tag_size_mm`,
`cell_size_mm`, `box_dims`). Voxel and twin configs also list every marker
with `id`, `face`, `normal` and `corners_mm`.

### Target size

For cuboids every dimension is a whole number of cells:

```text
cell_size  = tag_size / marker_pixels
axis_cells = 2 * border + N * marker_pixels + (N - 1) * margin
axis_mm    = axis_cells * cell_size
```

For example, `--grid 2x2x2 --dict 4x4_100 --tag-size 24` gives 4 mm cells and
a 60 mm cube. For voxel targets, each exposed voxel face is `voxel_size /
cell_size` cells with one centered marker.

Faces are numbered `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`. Each face's right and
down directions, seen from outside, satisfy `cross(right, down) = outward
normal`.

### How the detector works

```text
Frame -> grayscale -> ArUco detect -> keep the target's IDs -> visible faces
  -> 2D-3D correspondences -> PnP -> LM refinement -> Kalman update -> pose
```

- **Detection:** OpenCV ArUco with sub-pixel corner refinement, wide adaptive
  thresholding for printed surfaces, relaxed candidate filtering for oblique
  views, and fine bit sampling.
- **Multi-marker PnP:** all detected corners go into one solve, with
  `solvePnPRansac` (SQPNP, 3 px threshold) for 6 or more points, direct SQPNP
  for 4-5, then Levenberg-Marquardt refinement on the inliers. Markers on
  several faces remove the planar ambiguity of a single tag.
- **Error-state Kalman filter:** constant-velocity translation and a
  quaternion error state, with measurement noise adapted to reprojection
  error, tag count and inlier ratio, and Mahalanobis gating for outliers.

## Credits and citation

AprilFold is built on [aprilcube](https://github.com/younghyopark/aprilcube) by
Younghyo Park, released under the MIT License. AprilFold started from aprilcube
0.3.1 and keeps its git history. Marker generation and layout, voxel targets,
the browser designer, robot mounts and the detection pipeline come from
aprilcube. AprilFold adds the foldable single-color build, foldable voxel
targets and robot twins. See [NOTICE.md](../NOTICE.md).

If you use the targets or the detector in research, please cite aprilcube:

```bibtex
@software{park2026aprilcube,
  title={AprilCube: 3D-Printable Fiducial Targets for Reliable 6-DoF Pose Estimation},
  author={Park, Younghyo and Agrawal, Pulkit},
  year={2026},
  url={https://github.com/younghyopark/aprilcube},
}
```

The aprilcube technical report is available in the
[aprilcube repository](https://github.com/younghyopark/aprilcube/blob/main/docs/paper.pdf).
