# AprilFold

![AprilFold: a foldable single-color marker cube, exploded](https://raw.githubusercontent.com/muayad-chalabi/aprilfold/main/assets/aprilfold-thumbnail.png)

**Fiducial marker cubes for any 3D printer, and printed robot twins built from
them.**

AprilFold generates AprilTag/ArUco targets that print in one color at a time.
Each cube is a flat net that folds shut and locks itself, and every marker face
is a white plate with a black lattice that snaps on. No multi-material printer,
glue or tape is needed. A camera then tracks the cube's full 6-DoF pose.

On top of that, `aprilfold twin` turns a robot's URDF into a printed kinematic
twin, one foldable cube per link, joined by printed joints with hard stops at
the URDF limits. Move the twin by hand, track the cubes, and you have the joint
angles to teleoperate the real robot from vision alone.

AprilFold is built on [**aprilcube**](https://github.com/younghyopark/aprilcube)
by Younghyo Park. The marker layout, voxel targets and the detector come from
aprilcube; AprilFold adds the foldable single-color build and robot twins.

| Fold and assemble | Robot twin (SO-101, 1:1) |
|---|---|
| ![Folding and assembling a cube](https://raw.githubusercontent.com/muayad-chalabi/aprilfold/main/assets/fold_assembly.gif) | ![SO-101 and its printed twin](https://raw.githubusercontent.com/muayad-chalabi/aprilfold/main/assets/so101_twin.gif) |

## Install

```bash
pip install 'aprilfold[mono]'     # foldable cubes and voxel targets
pip install 'aprilfold[twin]'     # + robot twins
```

Python 3.10 or newer. From a checkout: `pip install -e '.[twin,test]'`.

## Foldable cube

```bash
aprilfold generate --grid 1x1x1 --dict 4x4_50 --tag-size 30 --mono -o my_cube
```

Print three files: `print_net.3mf` in any color, `print_white.3mf` in white and
`print_black.3mf` in black. Then:

1. Fold the `X` panels up, then the `Y` panels (their tabs slide into the `X`
   panels), then close the `+Z` lid until it clicks.
2. Lay each white plate on its panel; the panel is engraved with its tag ID.
3. Press each black lattice in: its barbed legs clamp lattice, plate and panel
   in one click.

Larger grids and voxel shapes (a 3x3x3 cube, a ring, a T) are built from cube
modules joined by printed rivets, so any shape assembles, loops included:

```bash
aprilfold generate --grid 3x3x3 --dict apriltag_36h11 --tag-size 30 --border-cell 1.5 --mono
aprilfold generate examples/foldable_ring.yaml        # also examples/foldable_t_shape.yaml
```

![Foldable targets](https://raw.githubusercontent.com/muayad-chalabi/aprilfold/main/assets/showcase.png)

## Robot twin

```bash
aprilfold twin so101.urdf.xacro -o so101_twin
aprilfold twin robot.urdf --scale 1.5 --last-link wrist -o arm_twin
```

The twin generator:

- fits the largest cube inside each link's convex hull;
- merges links joined by fixed joints into one body;
- builds one joint of each mimic group;
- designs a friction knuckle with hard stops for each revolute joint, and a bar
  and sleeve for each prismatic joint;
- checks the whole twin for self-collision across the joint ranges.

It writes one detector config per link, in that link's frame, and a
`twin.json` with the kinematic chain. Tested on the SO-101 and ElRobot arms
(1:1), the KUKA iiwa 14 (0.3x) and the Microduck biped (2x).
[models/so101_twin](models/so101_twin/README.md) is a complete example.

## Track

```python
import aprilfold

det = aprilfold.detector("my_cube/config.json", {"fx": 800, "fy": 800, "cx": 320, "cy": 240})
result = det.process_frame(frame)        # BGR image
if result["success"]:
    T = result["T"]                      # 4x4 pose in the camera frame, mm
```

Multi-marker PnP with RANSAC and an error-state Kalman filter, async
detection, world-frame poses and a 3D `viser` view are built in. The same
detector reads cube, voxel-target and twin-link configs.

## Documentation

- [User guide](docs/usage.md): printing and assembly, fit tuning, voxel
  targets, robot twins, the tracking API, and all options.
- [Changelog](CHANGELOG.md)

## Credits

AprilFold is a derivative of
[aprilcube](https://github.com/younghyopark/aprilcube) (MIT License) by
Younghyo Park, starting from aprilcube 0.3.1 with its git history kept. See
[NOTICE.md](NOTICE.md) for what comes from where. If you use the targets or
the detector in research, please cite aprilcube:

```bibtex
@software{park2026aprilcube,
  title={AprilCube: 3D-Printable Fiducial Targets for Reliable 6-DoF Pose Estimation},
  author={Park, Younghyo and Agrawal, Pulkit},
  year={2026},
  url={https://github.com/younghyopark/aprilcube},
}
```

## License

[MIT](LICENSE). Copyright Younghyo Park (aprilcube) and the AprilFold
contributors.
