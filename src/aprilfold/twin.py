"""Printed kinematic twin of a robot: one marker cube per link, joined by printed joints.

Each link of a URDF becomes a single-color cube module (see :mod:`aprilfold.mono`),
the largest cube that fits inside the link's convex hull.  The cubes are
connected by printed joint hardware placed exactly at the URDF joints, so
the twin moves like the robot.  Tracking the cubes with a camera gives every
link's pose, and with it the joint angles to drive the real robot.

Units: the URDF is in meters; everything here is in millimeters after
``scale`` is applied.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Robot description
# ---------------------------------------------------------------------------
@dataclass
class Joint:
    name: str
    type: str                 # revolute, continuous, prismatic, fixed
    parent: str
    child: str
    origin: np.ndarray        # 4x4, child joint frame in the parent link frame (mm)
    axis: np.ndarray          # unit vector in the joint frame
    lower: float = 0.0        # rad, or mm for prismatic
    upper: float = 0.0
    mimic: tuple[str, float, float] | None = None   # (joint, multiplier, offset): value = m * that joint + o

    @property
    def moves(self) -> bool:
        return self.type in ("revolute", "continuous", "prismatic")


@dataclass
class Link:
    name: str
    meshes: list = field(default_factory=list)   # trimesh.Trimesh in the link frame (mm)
    merged: list[str] = field(default_factory=list)   # links joined to this one by fixed joints

    @property
    def has_geometry(self) -> bool:
        return bool(self.meshes)


@dataclass
class Robot:
    name: str
    links: dict[str, Link]
    joints: list[Joint]
    mimics: list[Joint] = field(default_factory=list)   # joints left out because they follow another

    @property
    def root(self) -> str:
        children = {j.child for j in self.joints}
        roots = [name for name in self.links if name not in children]
        if len(roots) != 1:
            raise ValueError(f"robot must have exactly one root link, found {roots}")
        return roots[0]

    def child_joints(self, link: str) -> list[Joint]:
        return [j for j in self.joints if j.parent == link]

    def parent_joint(self, link: str) -> Joint | None:
        return next((j for j in self.joints if j.child == link), None)

    def forward(self, q: dict[str, float] | None = None) -> dict[str, np.ndarray]:
        """Link frames in the root frame for joint values ``q`` (rad or mm)."""
        q = q or {}
        frames = {self.root: np.eye(4)}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            for joint in self.child_joints(parent):
                frames[joint.child] = frames[parent] @ joint.origin @ joint_motion(joint, q.get(joint.name, 0.0))
                stack.append(joint.child)
        return frames


def joint_motion(joint: Joint, value: float) -> np.ndarray:
    mat = np.eye(4)
    if joint.type in ("revolute", "continuous"):
        mat[:3, :3] = rotation_about(joint.axis, value)
    elif joint.type == "prismatic":
        mat[:3, 3] = joint.axis * value
    return mat


def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _origin(elem: ET.Element | None, mm_per_unit: float) -> np.ndarray:
    mat = np.eye(4)
    if elem is None:
        return mat
    xyz = [float(v) for v in elem.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in elem.get("rpy", "0 0 0").split()]
    mat[:3, :3] = _rpy_matrix(*rpy)
    mat[:3, 3] = np.array(xyz) * mm_per_unit
    return mat


def expand_xacro(path: Path) -> str:
    """Run the ``xacro`` command on a .xacro file and return the URDF text."""
    exe = shutil.which("xacro")
    if exe is None:
        raise RuntimeError(
            f"{path.name} is a xacro file but the `xacro` command was not found. "
            "Source your ROS environment, or expand it yourself: xacro robot.urdf.xacro > robot.urdf"
        )
    result = subprocess.run([exe, str(path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"xacro failed on {path.name}:\n{result.stderr.strip()}")
    return result.stdout


def resolve_mesh_path(uri: str, base_dir: Path, package_paths: dict[str, Path],
                      mesh_dirs: list[Path] = ()) -> Path:
    """Resolve ``package://``, ``file://`` and relative mesh URIs.

    When the file is not where the URI says (a URDF copied without its
    meshes), each of ``mesh_dirs`` is searched for the URI's trailing path:
    ``a/b/c.stl`` is tried as ``dir/a/b/c.stl``, ``dir/b/c.stl`` and ``dir/c.stl``.
    """
    tail = uri.split("://", 1)[-1]
    found = None
    if uri.startswith("file://"):
        found = Path(tail)
    elif uri.startswith("package://"):
        package, _, rest = tail.partition("/")
        candidates = []
        if package in package_paths:
            candidates.append(Path(package_paths[package]))
        # A package directory above the robot file (source checkouts).
        for parent in [base_dir, *base_dir.parents]:
            if parent.name == package or (parent / package / "package.xml").exists():
                candidates.append(parent if parent.name == package else parent / package)
        for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
            if prefix:
                candidates.append(Path(prefix) / "share" / package)
        found = next((root / rest for root in candidates if (root / rest).exists()), None)
        if found is None and not mesh_dirs:
            raise FileNotFoundError(
                f"cannot find package '{package}' for {uri}; pass --package-path {package}=/path/to/{package} "
                "or --mesh-dir DIR"
            )
    else:
        found = Path(uri) if Path(uri).is_absolute() else base_dir / uri
    if found is not None and found.exists():
        return found
    parts = [part for part in Path(tail).parts if part not in ("/", "..", ".")]
    for directory in mesh_dirs:
        for k in range(len(parts)):
            candidate = Path(directory).expanduser() / Path(*parts[k:])
            if candidate.exists():
                return candidate
    if mesh_dirs:
        raise FileNotFoundError(f"mesh {uri} not found next to the robot file or in --mesh-dir")
    return found


def load_robot(path: str | os.PathLike, package_paths: dict[str, Path] | None = None,
               scale: float = 1.0, mesh_dirs: list[Path] = ()) -> Robot:
    """Load a URDF (or xacro) with meshes, in millimeters times ``scale``."""
    import trimesh

    path = Path(path).resolve()
    text = expand_xacro(path) if path.suffix == ".xacro" else path.read_text()
    root = ET.fromstring(text)
    mm = 1000.0 * scale
    package_paths = package_paths or {}

    def shapes(geoms):
        meshes = []
        for geom in geoms:
            g = geom.find("geometry")
            if g is None or len(g) == 0:
                continue
            shape = g[0]
            if shape.tag == "mesh":
                mesh_path = resolve_mesh_path(shape.get("filename"), path.parent, package_paths, mesh_dirs)
                mesh = trimesh.load(mesh_path, force="mesh")
                mesh.apply_scale(np.array([float(v) for v in shape.get("scale", "1 1 1").split()]))
            elif shape.tag == "box":
                mesh = trimesh.creation.box(extents=[float(v) for v in shape.get("size").split()])
            elif shape.tag == "cylinder":
                mesh = trimesh.creation.cylinder(radius=float(shape.get("radius")), height=float(shape.get("length")))
            elif shape.tag == "sphere":
                mesh = trimesh.creation.icosphere(radius=float(shape.get("radius")))
            else:
                continue
            mesh.apply_scale(mm)
            mesh.apply_transform(_origin(geom.find("origin"), mm))
            meshes.append(mesh)
        return meshes

    links: dict[str, Link] = {}
    for elem in root.findall("link"):
        link = Link(elem.get("name"))
        try:
            link.meshes = shapes(elem.findall("visual"))
        except FileNotFoundError:
            raise
        except Exception as exc:   # e.g. .dae visuals without pycollada: the collision shapes will do
            if not elem.findall("collision"):
                raise RuntimeError(f"link '{link.name}': cannot load its visual meshes ({exc})") from exc
        if not link.meshes:
            link.meshes = shapes(elem.findall("collision"))
        links[link.name] = link

    joints = []
    for elem in root.findall("joint"):
        jtype = elem.get("type")
        if jtype not in ("revolute", "continuous", "prismatic", "fixed"):
            raise ValueError(f"joint {elem.get('name')}: type '{jtype}' is not supported "
                             "(revolute, continuous, prismatic, fixed)")
        axis_elem = elem.find("axis")
        axis = np.array([float(v) for v in (axis_elem.get("xyz") if axis_elem is not None else "1 0 0").split()])
        limit = elem.find("limit")
        lower = float(limit.get("lower", 0)) if limit is not None else 0.0
        upper = float(limit.get("upper", 0)) if limit is not None else 0.0
        if jtype == "prismatic":
            lower, upper = lower * mm, upper * mm
        mimic = elem.find("mimic")
        if mimic is not None:
            mimic = (mimic.get("joint"), float(mimic.get("multiplier", 1)), float(mimic.get("offset", 0)))
        joints.append(Joint(
            name=elem.get("name"), type=jtype,
            parent=elem.find("parent").get("link"), child=elem.find("child").get("link"),
            origin=_origin(elem.find("origin"), mm),
            axis=axis / np.linalg.norm(axis), lower=lower, upper=upper, mimic=mimic,
        ))
    # Mimic multipliers in twin units (mm for prismatic joints, rad otherwise).
    unit = {j.name: mm if j.type == "prismatic" else 1.0 for j in joints}
    for j in joints:
        if j.mimic is not None and j.mimic[0] in unit:
            leader, mult, offset = j.mimic
            j.mimic = (leader, mult * unit[j.name] / unit[leader], offset * unit[j.name])
    return Robot(name=root.get("name", "robot"), links=links, joints=joints)


def _hull_volume(link: Link) -> float:
    import trimesh

    if not link.meshes:
        return 0.0
    return trimesh.convex.convex_hull(np.vstack([m.vertices for m in link.meshes])).volume


def drop_empty_links(robot: Robot, merge_fixed: bool = True) -> Robot:
    """Fold links joined by fixed joints into one rigid body, which gets one cube.

    A child behind a fixed joint is merged into its parent (its meshes move
    into the parent frame).  A geometry-less root (``world``) is removed and
    its largest fixed child becomes the root.  With ``merge_fixed`` off, only
    links without geometry are folded away and the others keep their own
    cubes, joined by printed struts.  Moving joints must connect links with
    geometry, since each needs a cube.
    """
    links = {n: Link(n, list(link.meshes), list(link.merged)) for n, link in robot.links.items()}
    joints = [Joint(**{**j.__dict__}) for j in robot.joints]
    changed = True
    while changed:
        changed = False
        roots = [n for n in links if not any(j.child == n for j in joints)]
        for root in roots:
            fixed = [j for j in joints if j.parent == root and j.type == "fixed"]
            if links[root].has_geometry or not fixed:
                continue
            # Empty root: its largest fixed child becomes the root.
            joint = max(fixed, key=lambda j: _hull_volume(links[j.child]))
            joints.remove(joint)
            for other in joints:
                if other.parent == root:
                    other.parent = joint.child
                    other.origin = np.linalg.inv(joint.origin) @ other.origin
            del links[root]
            changed = True
            break
        if changed:
            continue
        for joint in list(joints):
            parent, child = links[joint.parent], links[joint.child]
            if joint.type != "fixed" or (child.has_geometry and not merge_fixed):
                continue
            joints.remove(joint)
            for mesh in child.meshes:
                parent.meshes.append(mesh.copy().apply_transform(joint.origin))
            parent.merged += [child.name, *child.merged]
            for other in joints:
                if other.parent == child.name:
                    other.parent = parent.name
                    other.origin = joint.origin @ other.origin
            del links[child.name]
            changed = True
            break
    for joint in joints:
        for name in (joint.parent, joint.child):
            if not links[name].has_geometry:
                raise ValueError(f"link '{name}' has no visual or collision geometry, so it cannot carry a cube")
    return Robot(name=robot.name, links=links, joints=joints, mimics=list(robot.mimics))


def drop_mimic(robot: Robot) -> Robot:
    """Keep one joint of each mimic group; leave out the others, with everything beyond them.

    Joints that mimic another move together with it, so the twin needs only
    one of them: the one whose link is largest, which gets the best cube.
    Every left-out joint is kept in ``robot.mimics``, re-expressed as
    ``value = multiplier * kept joint + offset``.
    """
    by_name = {j.name: j for j in robot.joints}
    groups: dict[str, list[Joint]] = {}
    for joint in robot.joints:
        if joint.mimic is not None and joint.mimic[0] in by_name:
            groups.setdefault(joint.mimic[0], [by_name[joint.mimic[0]]]).append(joint)
    keep_joints, mimics = [], []
    for leader, members in groups.items():
        kept = max(members, key=lambda j: _hull_volume(robot.links[j.child]))
        # kept = mk * leader + ok  ->  leader = (kept - ok) / mk
        mk, ok = (1.0, 0.0) if kept.name == leader else kept.mimic[1:]
        for joint in members:
            if joint is kept:
                continue
            m, o = (1.0, 0.0) if joint.name == leader else joint.mimic[1:]
            mimics.append(Joint(**{**joint.__dict__, "mimic": (kept.name, m / mk, o - m * ok / mk)}))
        keep_joints.append(kept.name)
    gone: set[str] = set()
    for joint in mimics:
        gone |= _subtree(robot, joint.child)
    joints = [Joint(**{**j.__dict__, "mimic": None}) if j.name in keep_joints else j
              for j in robot.joints if j.child not in gone]
    return Robot(name=robot.name, links={n: link for n, link in robot.links.items() if n not in gone},
                 joints=joints, mimics=robot.mimics + mimics)


def cut_after(robot: Robot, tip: str) -> Robot:
    """Keep ``tip`` and drop every link beyond it (its descendants) with their joints."""
    if tip not in robot.links:
        raise ValueError(f"no link '{tip}' in {robot.name}; links: {', '.join(robot.links)}")
    gone: set[str] = set()
    stack = [j.child for j in robot.joints if j.parent == tip]
    while stack:
        name = stack.pop()
        gone.add(name)
        stack += [j.child for j in robot.joints if j.parent == name]
    return Robot(name=robot.name, links={n: link for n, link in robot.links.items() if n not in gone},
                 joints=[j for j in robot.joints if j.child not in gone], mimics=list(robot.mimics))


# ---------------------------------------------------------------------------
# Largest cube inside a convex hull
# ---------------------------------------------------------------------------
@dataclass
class Hull:
    """Convex hull as half-spaces ``normals @ x <= offsets`` (duplicate planes merged)."""
    normals: np.ndarray
    offsets: np.ndarray
    mesh: object

    @classmethod
    def of(cls, points: np.ndarray) -> "Hull":
        import trimesh

        mesh = trimesh.convex.convex_hull(points)
        # Curved meshes give thousands of facets.  Facets whose normals agree to about
        # 6 degrees share one plane, pushed out to the farthest hull vertex, so the
        # hull grows by a fraction of a millimeter and the LPs stay small.
        normals = mesh.face_normals
        _, group = np.unique(np.round(normals * 10.0).astype(int), axis=0, return_inverse=True)
        merged = np.zeros((group.max() + 1, 3))
        np.add.at(merged, group.ravel(), normals)
        merged /= np.linalg.norm(merged, axis=1, keepdims=True)
        return cls(merged, (mesh.vertices @ merged.T).max(axis=0), mesh)


def _cube_lp(hull: Hull, rot: np.ndarray, extra: list[tuple[np.ndarray, float]] = (),
             edge: float | None = None, weights=None) -> tuple[float, np.ndarray, float]:
    """Largest cube with orientation ``rot`` inside the hull and the extra half-spaces.

    ``extra`` holds (n, d) meaning the whole cube satisfies ``n @ x <= d``;
    those are hard.  With ``edge`` given, the cube size is fixed and the hull
    may be exceeded: the LP instead minimises how far it sticks out, and with
    ``weights`` the extra half-spaces may be crossed too, at that cost per mm.
    Returns (edge, center, hull overshoot).
    """
    from scipy.optimize import linprog

    def reach(n):
        return np.abs(np.atleast_2d(n) @ rot).sum(axis=1) / 2.0

    hard = [np.append(n, reach(n)[0]) for n, _ in extra]
    hard_b = [d for _, d in extra]
    if edge is None:
        a_ub = np.vstack([np.column_stack([hull.normals, reach(hull.normals)]), *hard]) if hard else \
            np.column_stack([hull.normals, reach(hull.normals)])
        b_ub = np.concatenate([hull.offsets, hard_b])
        res = linprog([0, 0, 0, -1], A_ub=a_ub, b_ub=b_ub, bounds=[(None, None)] * 3 + [(0, None)], method="highs")
        if not res.success:
            return 0.0, np.zeros(3), 0.0
        return float(res.x[3]), res.x[:3], 0.0
    # Fixed edge: variables (c, t, s): hull rows relaxed by t >= 0, extra rows by
    # s_k >= 0 at a cost of weights[k] each (hard when no weights are given).
    k = len(extra)
    w = np.full(k, 1e4) if weights is None else np.asarray(weights, dtype=float)
    a_hull = np.hstack([hull.normals, -np.ones((len(hull.normals), 1)), np.zeros((len(hull.normals), k))])
    b_hull = hull.offsets - edge * reach(hull.normals)
    a_extra = np.hstack([np.array([n for n, _ in extra]).reshape(-1, 3), np.zeros((k, 1)), -np.eye(k)])
    b_extra = np.array([d - edge * reach(n)[0] for n, d in extra])
    a_ub, b_ub = np.vstack([a_hull, a_extra]), np.concatenate([b_hull, b_extra])
    cost = np.concatenate([[0, 0, 0, 1], w])
    bounds = [(None, None)] * 3 + [(0, None)] * (1 + k)
    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return 0.0, np.zeros(3), np.inf
    # Many centers cost the same; take the one nearest the hull's middle, so the
    # cube overhangs its link evenly.  Extra variables |c - mid|.
    mid = hull.mesh.centroid
    a2 = np.hstack([a_ub, np.zeros((len(a_ub), 3))])
    cap = np.concatenate([cost, np.zeros(3)])[None, :]
    absdev = np.vstack([np.hstack([sign * np.eye(3), np.zeros((3, 1 + k)), -np.eye(3)]) for sign in (1, -1)])
    res2 = linprog(np.concatenate([np.zeros(4 + k), np.ones(3)]), A_ub=np.vstack([a2, cap, absdev]),
                   b_ub=np.concatenate([b_ub, [res.fun + 0.5], mid, -mid]),
                   bounds=bounds + [(0, None)] * 3, method="highs")
    best = res2 if res2.success else res
    return edge, best.x[:3], float(best.x[3])


def _random_rotations(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    quats = rng.normal(size=(n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    rots = []
    for w, x, y, z in quats:
        rots.append(np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]))
    return rots


def largest_cube(hull: Hull, extra: list[tuple[np.ndarray, float]] = (), samples: int = 300,
                 seed: int = 0, prefer_aligned: float = 0.97) -> tuple[float, np.ndarray]:
    """Edge and 4x4 pose of the largest cube inside ``hull`` (and ``extra`` half-spaces).

    Orientations are searched; the link-aligned cube wins unless a turned one
    is more than ``1 / prefer_aligned`` larger, which keeps cubes square to
    their link where it costs little.
    """
    aligned_edge, aligned_center, _ = _cube_lp(hull, np.eye(3), extra)
    best = (aligned_edge, aligned_center, np.eye(3))
    for rot in _random_rotations(samples, np.random.default_rng(seed)):
        edge, center, _ = _cube_lp(hull, rot, extra)
        if edge > best[0]:
            best = (edge, center, rot)
    step = 0.15
    while step > 1e-3:
        improved = False
        for axis in np.eye(3):
            for sign in (1, -1):
                rot = rotation_about(axis, sign * step) @ best[2]
                edge, center, _ = _cube_lp(hull, rot, extra)
                if edge > best[0] + 1e-6:
                    best, improved = (edge, center, rot), True
        if not improved:
            step /= 2
    edge, center, rot = best
    if aligned_edge >= prefer_aligned * edge:
        edge, center, rot = aligned_edge, aligned_center, np.eye(3)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rot, center
    return edge, pose


def place_cube(hull: Hull, edge: float, rot: np.ndarray, extra: list[tuple[np.ndarray, float]] = (),
               weights=None) -> tuple[np.ndarray, float]:
    """Pose of a cube of fixed ``edge`` that sticks out of ``hull`` as little as possible.

    With ``weights``, the ``extra`` half-spaces are soft (cost per mm crossed).
    """
    _, center, overshoot = _cube_lp(hull, rot, extra, edge=edge, weights=weights)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rot, center
    return pose, overshoot


# ---------------------------------------------------------------------------
# Cube planning
# ---------------------------------------------------------------------------
_UNIT_CORNERS = np.array([[i, j, k] for i in (-0.5, 0.5) for j in (-0.5, 0.5) for k in (-0.5, 0.5)])


def boxes_intersect(edge_a: float, pose_a: np.ndarray, edge_b: float, pose_b: np.ndarray,
                    margin: float = 0.0) -> bool:
    """Separating-axis test for two cubes (pose = 4x4 center and orientation)."""
    pa = (pose_a[:3, :3] @ (_UNIT_CORNERS * edge_a).T).T + pose_a[:3, 3]
    pb = (pose_b[:3, :3] @ (_UNIT_CORNERS * edge_b).T).T + pose_b[:3, 3]
    axes = [pose_a[:3, i] for i in range(3)] + [pose_b[:3, i] for i in range(3)]
    axes += [np.cross(u, v) for u in axes[:3] for v in axes[3:]]
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            continue
        axis = axis / norm
        if (pa @ axis).max() + margin < (pb @ axis).min() or (pb @ axis).max() + margin < (pa @ axis).min():
            return False
    return True


def sweep_values(joint: Joint, count: int | None = None) -> np.ndarray:
    """Joint values across the range: about every 4 degrees (or 2 mm), at least 9."""
    if joint.type == "continuous" or (joint.type == "revolute" and joint.upper - joint.lower >= 2 * np.pi - 1e-6):
        return np.linspace(-np.pi, np.pi, count or 91)
    if joint.type == "revolute":
        n = count or max(9, int(np.ceil(np.degrees(joint.upper - joint.lower) / 4.0)) + 1)
        return np.linspace(joint.lower, joint.upper, n)
    if joint.type == "prismatic":
        n = count or max(9, int(np.ceil((joint.upper - joint.lower) / 2.0)) + 1)
        return np.linspace(joint.lower, joint.upper, n)
    return np.zeros(1)


@dataclass
class MarkerLayout:
    dict_name: str
    border_cells: float
    tag_size: float
    cell_size: float
    marker_pixels: int


@dataclass
class LinkCube:
    link: str
    edge: float
    pose: np.ndarray          # cube center and orientation in the link frame (mm)
    layout: MarkerLayout
    overshoot: float = 0.0    # how far the cube sticks out of the link's hull (mm)
    fitted_edge: float = 0.0  # largest edge that fitted inside the hull


def marker_layout(edge: float, dict_name: str, mono_cfg) -> MarkerLayout | None:
    """Smallest border that leaves room for the snap legs on a cube of this edge."""
    from aprilfold import mono
    from aprilfold.generate import DICT_MAP, TagPatternGenerator

    mp = TagPatternGenerator(DICT_MAP[dict_name]).marker_pixels
    for border in np.arange(1.0, 8.01, 0.5):
        cell = edge / (mp + 2 * border)
        tag = cell * mp
        try:
            mono._check_dims(edge / 2.0, tag, cell, border, mono_cfg)
        except ValueError:
            continue
        return MarkerLayout(dict_name, float(border), tag, cell, mp)
    return None


def min_module_edge(dict_name: str, mono_cfg, start: float = 15.0) -> float:
    """Smallest cube edge (0.5 mm steps) that makes a working mono module."""
    from aprilfold import mono

    edge = start
    while edge < 200.0:
        layout = marker_layout(edge, dict_name, mono_cfg)
        if layout is not None:
            dims, legs = mono._check_dims(edge / 2.0, layout.tag_size, layout.cell_size, layout.border_cells, mono_cfg)
            try:
                mono._build_module(dims, legs, np.eye(3), {}, {})
                return edge
            except ValueError:
                pass
        edge += 0.5
    raise ValueError(f"no single-color module fits {dict_name} markers below 200 mm")


def _separation(joint: Joint, a: LinkCube, a_to_parent: np.ndarray, b: LinkCube, b_to_child: np.ndarray,
                gap: float) -> tuple[tuple[np.ndarray, float], tuple[np.ndarray, float]]:
    """Half-spaces (in a's frame, in b's frame) that keep two cubes apart while ``joint`` moves.

    ``a`` rides on the joint's parent side (``a_to_parent``: its frame in the
    parent link frame), ``b`` on the child side (``b_to_child``).
    Revolute: a plane across the axis, which keeps them apart at every angle.
    Prismatic and fixed: a plane along the line between them.
    """
    to_b = joint.origin @ b_to_child
    jr = joint.origin[:3, :3]
    pa = (a_to_parent @ a.pose)[:3, 3]
    pb = (to_b @ b.pose)[:3, 3]          # b's center in the parent frame at q = 0
    if joint.type in ("revolute", "continuous"):
        n = jr @ joint.axis
    else:
        n = pb - pa
        if joint.type == "prismatic":
            axis = jr @ joint.axis
            n = n - (n @ axis) * axis
    n = n / np.linalg.norm(n)
    if (pb - pa) @ n < 0:
        n = -n
    mid = (pa + pb) / 2.0 @ n
    # n @ (R x + t) <= mid - gap / 2 for a, and >= mid + gap / 2 for b.
    a_hs = (a_to_parent[:3, :3].T @ n, mid - gap / 2.0 - n @ a_to_parent[:3, 3])
    b_hs = (-to_b[:3, :3].T @ n, -(mid + gap / 2.0) + n @ to_b[:3, 3])
    return a_hs, b_hs


RESERVE_RADIUS = 17.0     # knuckle discs, bolt head and nut fit inside this radius
RESERVE_HALF_LENGTH = 18.0   # and this far along the axis either side of the joint origin


def _joint_line(joint: Joint, side: str) -> tuple[np.ndarray, np.ndarray]:
    """(point, axis direction) of a joint in the parent or child link frame."""
    if side == "parent":
        return joint.origin[:3, 3].copy(), joint.origin[:3, :3] @ joint.axis
    return np.zeros(3), joint.axis.copy()


def _reservation_options(joint: Joint, side: str, center: np.ndarray):
    """Ways to keep a cube clear of a revolute joint's knuckle stack, as (mode, sign, half-space).

    ``axial``: beyond the reserved length, on either side of the origin along
    the axis.  ``radial``: behind a plane tangent to the reserved cylinder,
    toward the cube's free center, away from it, or to either side.
    """
    point, axis = _joint_line(joint, side)
    out = []
    for sign in (1.0, -1.0):
        n = sign * axis
        out.append(("axial", sign, (-n, -(n @ point + RESERVE_HALF_LENGTH))))
    off = center - point
    off = off - (off @ axis) * axis
    if np.linalg.norm(off) < 1e-6:
        off = np.cross(axis, [1.0, 0.0, 0.0])
        if np.linalg.norm(off) < 1e-6:
            off = np.cross(axis, [0.0, 1.0, 0.0])
    u = off / np.linalg.norm(off)
    w = np.cross(axis, u)
    for n in (u, -u, w, -w):
        out.append(("radial", 0.0, (-n, -(n @ point + RESERVE_RADIUS))))
    return out


def _keep_clear(point: np.ndarray, axis: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, float]:
    """Half-space between a reserved knuckle cylinder and a cube around ``center``.

    Whichever of its side (radial) or end (axial) faces leaves the cube more room.
    """
    off = center - point
    along = off @ axis
    radial = off - along * axis
    r = np.linalg.norm(radial)
    if abs(along) - RESERVE_HALF_LENGTH > r - RESERVE_RADIUS or r < 1e-6:
        n = (np.sign(along) or 1.0) * axis
        return -n, -(n @ point + RESERVE_HALF_LENGTH)
    n = radial / r
    return -n, -(n @ point + RESERVE_RADIUS)


def _placement_cost(hull: Hull, edge: float, pose: np.ndarray, extra, weights) -> float:
    """Hull overshoot plus weighted crossing of ``extra`` half-spaces, for a cube at ``pose``."""
    corners = (pose[:3, :3] @ (_UNIT_CORNERS * edge).T).T + pose[:3, 3]
    over = max(0.0, float((corners @ hull.normals.T - hull.offsets).max()))
    for (n, d), w in zip(extra, weights):
        over += w * max(0.0, float((corners @ n).max() - d))
    return over


HARD_WEIGHT = 8.0   # cost per mm of an oversized cube crossing its own knuckle or separation planes
STRICT_WEIGHT = 200.0   # ... for a joint whose knuckle found no room the first time, and for the
                        # planes that keep two cubes apart
NEAR_WEIGHT = 4.0   # ... or a neighbor's knuckle room, shared over its sampled angles (vs. 1 per mm out of the hull)


def _neighbor_frames(robot: Robot, link: str, skip: Joint):
    """(neighbor, poses of its frame in ``link``'s frame across the joint range) for every joint but ``skip``."""
    for joint in robot.joints:
        if joint is skip or link not in (joint.parent, joint.child):
            continue
        mats = [joint.origin @ joint_motion(joint, v) for v in sweep_values(joint)]
        if joint.parent == link:
            yield joint.child, mats
        else:
            yield joint.parent, [np.linalg.inv(m) for m in mats]


def _subtree(robot: Robot, link: str) -> set[str]:
    out, stack = set(), [link]
    while stack:
        name = stack.pop()
        out.add(name)
        stack += [j.child for j in robot.child_joints(name)]
    return out


def _first_clash(robot: Robot, cubes: dict[str, LinkCube], adjacent: set, separated: set,
                 gaps: dict[str, float], robot_clear) -> tuple[Joint, str, str] | None:
    """(joint, a, b): cubes a (parent side) and b (child side) meet as that one joint moves.

    Neighbors come first, then links further apart, but only where the robot's own
    links are clear (``robot_clear``).  The other joints stay at zero.
    """
    for neighbors_only in (True, False):
        for joint in robot.joints:
            if neighbors_only:
                pairs = [(joint.parent, joint.child)]
            else:
                moving = _subtree(robot, joint.child)
                pairs = [(a, b) for a in robot.links if a not in moving for b in moving
                         if frozenset((a, b)) not in adjacent]
            pairs = [(a, b) for a, b in pairs if (joint.name, a, b) not in separated]
            if not pairs:
                continue
            for value in sweep_values(joint):
                frames = robot.forward({joint.name: value})
                for a, b in pairs:
                    ca, cb = cubes[a], cubes[b]
                    if (boxes_intersect(ca.edge, frames[a] @ ca.pose, cb.edge, frames[b] @ cb.pose,
                                        margin=gaps.get(joint.name, 2.0))
                            and (neighbors_only or robot_clear(frames, a, b))):
                        return joint, a, b
    return None


def plan_cubes(robot: Robot, dict_name: str, mono_cfg, gaps: dict[str, float] | None = None,
               samples: int = 300, strict: set[str] = frozenset()) -> dict[str, LinkCube]:
    """Fit one cube per link, clear of the joint hardware and of each other.

    Oversized cubes may cross the joints' reserved knuckle room a little when
    nothing else fits; for the ``strict`` joints they practically may not.
    """
    gaps = gaps or {}
    min_edge = min_module_edge(dict_name, mono_cfg)
    hulls = {name: Hull.of(np.vstack([m.vertices for m in link.meshes])) for name, link in robot.links.items()}
    extra: dict[str, list] = {name: [] for name in robot.links}
    near: dict[str, list] = {name: [] for name in robot.links}   # neighbors' reservations, swept
    near_owner: dict[str, list] = {name: [] for name in robot.links}   # ... and whose they are

    # Reserve room for every revolute joint's knuckle stack around its origin.
    free = {name: _cube_lp(hulls[name], np.eye(3))[1] for name in robot.links}

    def value(name, hs):
        """How good a cube this link still gets with one more reservation (link-aligned, for speed)."""
        rows = extra[name] + [hs]
        edge = _cube_lp(hulls[name], np.eye(3), rows)[0]
        if edge >= min_edge:
            return min_edge + 1e-3 * edge
        # Too small: the cost of the oversized cube sticking out or crossing its reservations.
        pose, _ = place_cube(hulls[name], min_edge, np.eye(3), rows, [HARD_WEIGHT] * len(rows))
        return min_edge - _placement_cost(hulls[name], min_edge, pose, rows, [HARD_WEIGHT] * len(rows))

    # Each joint picks how its cubes keep clear given the others' picks; a few
    # rounds let early picks adapt to later ones.
    revolute = [j for j in robot.joints if j.type in ("revolute", "continuous")]
    chosen: dict[str, tuple] = {}
    for _ in range(3):
        for joint in revolute:
            if joint.name in chosen:
                old_p, old_c = chosen[joint.name]
                extra[joint.parent] = [hs for hs in extra[joint.parent] if hs is not old_p]
                extra[joint.child] = [hs for hs in extra[joint.child] if hs is not old_c]
            sides = {}
            for side, link in (("parent", joint.parent), ("child", joint.child)):
                sides[side] = [(mode, sign, hs, value(link, hs))
                               for mode, sign, hs in _reservation_options(joint, side, free[link])]
            options = [(pv + cv, p_hs, c_hs)
                       for p_mode, p_sign, p_hs, pv in sides["parent"]
                       for c_mode, c_sign, c_hs, cv in sides["child"]
                       if not (p_mode == c_mode == "axial" and p_sign == c_sign)]   # stack between them
            _, p_hs, c_hs = max(options, key=lambda o: o[0])
            extra[joint.parent].append(p_hs)
            extra[joint.child].append(c_hs)
            chosen[joint.name] = (p_hs, c_hs)
    for joint in revolute:
        # The neighbors' cubes, moving through their own joints, must keep out of it too.
        for side, which in ((joint.parent, "parent"), (joint.child, "child")):
            point, axis = _joint_line(joint, which)
            for other, to_side in _neighbor_frames(robot, side, skip=joint):
                for mat in to_side:
                    back = np.linalg.inv(mat)
                    near[other].append(_keep_clear(back[:3, :3] @ point + back[:3, 3], back[:3, :3] @ axis,
                                                   free[other]))
                    near_owner[other].append(joint.name)

    zero = robot.forward()
    adjacent = {frozenset((j.parent, j.child)) for j in robot.joints}
    real = {}

    def robot_clear(frames, a, b):
        if not real:
            real.update(_robot_solids(robot))
        return _links_clear(real, frames, a, b)

    separated: set[tuple[str, str, str]] = set()
    cubes: dict[str, LinkCube] = {}
    dirty = set(robot.links)
    for _ in range(4 * len(robot.links) ** 2):
        for name in dirty:
            hull = hulls[name]
            edge, pose = largest_cube(hull, extra[name], samples=samples)
            fitted, overshoot = edge, 0.0
            if edge < min_edge:
                # Sticking out of its hull, the cube must also keep out of the
                # neighbors' knuckles, which the hull no longer guarantees.
                # Both are soft: on a link much smaller than the cube they may not all
                # hold, and the cube should then give a little rather than fly away.
                strict_rows = {id(hs) for j in strict if j in chosen for hs in chosen[j]}
                reserve_rows = {id(hs) for pair in chosen.values() for hs in pair}
                # Knuckle room may give a little; the planes between cubes may not.
                weights = [STRICT_WEIGHT if id(hs) in strict_rows else HARD_WEIGHT if id(hs) in reserve_rows
                           else STRICT_WEIGHT for hs in extra[name]]
                # A neighbor's knuckle comes as one row per sampled angle: they share one weight.
                counts = {owner: near_owner[name].count(owner) for owner in set(near_owner[name])}
                weights += [(STRICT_WEIGHT if owner in strict else NEAR_WEIGHT) / counts[owner]
                            for owner in near_owner[name]]
                pose, overshoot = place_cube(hull, min_edge, pose[:3, :3], extra[name] + near[name], weights)
                edge = min_edge
            layout = marker_layout(edge, dict_name, mono_cfg)
            cubes[name] = LinkCube(name, edge, pose, layout, overshoot, fitted)
        clash = _first_clash(robot, cubes, adjacent, separated, gaps, robot_clear)
        if clash is None:
            return cubes
        joint, a, b = clash
        separated.add((joint.name, a, b))
        a_to_parent = np.linalg.inv(zero[joint.parent]) @ zero[a]
        b_to_child = np.linalg.inv(zero[joint.child]) @ zero[b]
        a_hs, b_hs = _separation(joint, cubes[a], a_to_parent, cubes[b], b_to_child, gaps.get(joint.name, 2.0))
        extra[a].append(a_hs)
        extra[b].append(b_hs)
        dirty = {a, b}
    raise ValueError("could not separate the link cubes")


# ---------------------------------------------------------------------------
# Joint hardware
# ---------------------------------------------------------------------------
DISC_T = 5.0          # knuckle disc thickness
DISC_GAP = 0.4        # between the two knuckle discs
BOLT_R = 4.5          # bolt shank radius
BOLT_FLAT = 0.8       # depth of the D-flat that keys the bolt to the nut-side disc
BORE_CLEAR = 0.25     # bore clearance around the shank
HEAD_R, HEAD_T = 7.0, 3.0
NUT_R, NUT_T = 9.0, 6.0
THREAD_E = 0.6        # eccentric "round" thread: depth = 2 * THREAD_E
THREAD_PITCH = 2.5
THREAD_CLEAR = 0.3
STOP_PIN_R = 1.6
STOP_DEPTH = 1.8
STOP_CLEAR = 0.3
STRUT_W = 8.0         # strut width (and lug width)
MOUNT_SLAB = 1.5      # strut foot on the cube face
DISC_RADII = (16.0, 14.0, 12.0)


def _m():
    from manifold3d import CrossSection, Manifold, OpType
    return CrossSection, Manifold, OpType


def _place(part, mat: np.ndarray):
    return part.transform(np.asarray(mat)[:3, :])


def _z_to(axis: np.ndarray) -> np.ndarray:
    """Rotation taking +Z onto ``axis``."""
    z = np.array([0.0, 0.0, 1.0])
    a = axis / np.linalg.norm(axis)
    v, c = np.cross(z, a), z @ a
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _hull(points: np.ndarray):
    _, Manifold, _ = _m()
    return Manifold.hull_points(np.asarray(points, dtype=float))


def _disc(radius: float, z0: float, z1: float, segments: int = 96):
    _, Manifold, _ = _m()
    return Manifold.cylinder(z1 - z0, radius, circular_segments=segments).translate((0.0, 0.0, z0))


def _box_points(center: np.ndarray, axes: list[np.ndarray], half: list[float]) -> np.ndarray:
    return np.array([center + sum(s * h * a for s, h, a in zip(signs, half, axes))
                     for signs in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)])


@dataclass
class CubeSolid:
    """A link cube as seen from a joint frame (mm)."""
    edge: float
    pose: np.ndarray          # 4x4 in the joint frame

    def manifold(self, shrink: float = 0.0):
        _, Manifold, _ = _m()
        e = self.edge - 2 * shrink
        return _place(Manifold.cube((e, e, e), center=True), self.pose)

    def mount(self, toward: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        """(face index, face center, outward normal) of the face looking at ``toward``."""
        center = self.pose[:3, 3]
        normals = [s * self.pose[:3, i] for i in range(3) for s in (1, -1)]
        scores = [n @ (toward - center) for n in normals]
        k = int(np.argmax(scores))
        return k, center + normals[k] * self.edge / 2.0, normals[k]


FACE_ORDER = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]   # index k -> cube-local face name


def _strut(cube: CubeSolid, z0: float, z1: float, radius: float, style: str, turn: float = 0.0):
    """Arm from a knuckle disc (axis z, between z0 and z1) to a face of the cube.

    ``direct``: one straight hull from a lug on the disc rim to the face that
    looks at the lug.  ``dogleg``: out along the disc's plane to below the cube's
    center, then straight into the face that looks at that point; ``dogleg_out``
    rises a little further out, clear of the nut; ``dogleg_wide`` rises beside
    the cube into its outer side face; ``riser`` runs along the axis just
    outside the disc to the cube's height, then into the face that looks at
    the axis.  ``fork``:
    two arms to opposite edges of a face across the axis (for cubes on the axis).
    ``turn`` (rad) swings the lug and elbow around the axis away from the
    cube's direction, to pass beside a neighbor or through the gap its
    motion leaves.
    """
    center = cube.pose[:3, 3]
    perp = np.array([center[0], center[1], 0.0])
    u = perp / np.linalg.norm(perp) if np.linalg.norm(perp) > 1e-6 else np.array([1.0, 0.0, 0.0])
    u = np.array([u[0] * np.cos(turn) - u[1] * np.sin(turn), u[0] * np.sin(turn) + u[1] * np.cos(turn), 0.0])
    t = np.cross([0.0, 0.0, 1.0], u)
    zm = (z0 + z1) / 2.0
    zaxis = np.array([0.0, 0.0, 1.0])
    lug_c = np.array([*(u * (radius + 1.5))[:2], zm])
    lug = _box_points(lug_c, [u, t, zaxis], [3.0, STRUT_W / 2, (z1 - z0) / 2])

    def foot(toward, side=None):
        """Pad on the face that looks at ``toward``, right where the strut meets it."""
        k, face_c, normal = cube.mount(toward)
        a, b = [cube.pose[:3, i] for i in range(3) if abs(cube.pose[:3, i] @ normal) < 0.5]
        side = side or min(0.5 * cube.edge, 3 * STRUT_W)
        room = cube.edge / 2 - side / 2 - 1.0
        offset = toward - face_c
        center = face_c + np.clip(offset @ a, -room, room) * a + np.clip(offset @ b, -room, room) * b
        return k, _box_points(center + normal * MOUNT_SLAB / 2, [a, b, normal], [side / 2, side / 2, MOUNT_SLAB / 2])

    if style == "direct":
        k, slab = foot(lug_c)
        return _hull(np.vstack([lug, slab])), k
    if style == "fork":
        # Two arms from opposite sides of the rim to opposite edges of the face that looks
        # along the axis, leaving the axis free for the bolt, nut and fingers.
        k, face_c, normal = cube.mount(np.array([center[0], center[1], zm]))
        if abs(normal[2]) < 0.9:
            return _hull(lug) - _hull(lug), k   # only for faces across the axis
        a = next(cube.pose[:3, i] for i in range(3) if abs(cube.pose[:3, i] @ normal) < 0.5)
        b = np.cross(normal, a)
        reach = cube.edge / 2 - STRUT_W / 2 - 1.0
        arms = []
        for sign in (1, -1):
            rim = np.array([*(sign * a * (radius + 1.5))[:2], zm])
            rim_box = _box_points(rim, [sign * a, b, np.array([0, 0, 1.0])], [3.0, STRUT_W / 2, (z1 - z0) / 2])
            edge = face_c + sign * a * reach + normal * MOUNT_SLAB / 2
            edge_box = _box_points(edge, [a, b, normal], [STRUT_W / 2, STRUT_W / 2, MOUNT_SLAB / 2])
            arms.append(_hull(np.vstack([rim_box, edge_box])))
        return arms[0] + arms[1], k
    if style == "riser":
        # Straight along the axis, just outside the disc, to the cube's height, then out
        # into the face that looks back at the axis: stays inside whatever sweeps around it.
        elbow_c = np.array([*(u * (radius + 1.5))[:2], center[2]])
        elbow = _box_points(elbow_c, [u, t, zaxis], [3.0, STRUT_W / 2, (z1 - z0) / 2])
        k, slab = foot(elbow_c, side=STRUT_W + 2.0)
        return _hull(np.vstack([lug, elbow])) + _hull(np.vstack([elbow, slab])), k
    rho = max(np.linalg.norm(perp), radius + 1.5)
    if style == "dogleg_out":
        # Rise just outside the nut's (and a hand's) room around the axis.
        rho = max(rho, NUT_R + 5.0 + STRUT_W / 2 + 2.0)   # its foot pad is 1 mm wider than the strut
    elif style == "dogleg_wide":
        # Rise beside the cube and land on its outer side face, for cubes sitting on the axis.
        corners = (cube.pose[:3, :3] @ (_UNIT_CORNERS * cube.edge).T).T + center
        rho = max(rho, NUT_R + 5.0 + STRUT_W / 2 + 2.0, (corners @ u).max() + STRUT_W / 2 + MOUNT_SLAB + 1.0)
    elbow_c = np.array([*(u * rho)[:2], zm])
    elbow = _box_points(elbow_c, [u, t, zaxis], [STRUT_W / 2, STRUT_W / 2, (z1 - z0) / 2])
    # A pad just wider than the strut keeps the rising leg straight.
    k, slab = foot(elbow_c, side=STRUT_W + 2.0)
    return _hull(np.vstack([lug, elbow])) + _hull(np.vstack([elbow, slab])), k


def _thread(radius: float, length: float, grow: float = 0.0):
    """Eccentric round thread along +z: a circle offset from the axis, twisted.

    Its outer radius is ``radius``, so it passes through a bore of that size.
    """
    CrossSection, _, _ = _m()
    section = CrossSection.circle(radius - THREAD_E + grow, 48).translate((THREAD_E, 0.0))
    turns = length / THREAD_PITCH
    return section.extrude(length, n_divisions=max(8, int(turns * 24)), twist_degrees=-360.0 * turns)


def _d_section(radius: float, grow: float = 0.0):
    """Circle with a flat: the bolt shank, and the bore that keys it."""
    CrossSection, _, _ = _m()
    r = radius + grow
    flat_at = radius - BOLT_FLAT + grow
    return CrossSection.circle(r, 48) ^ CrossSection.square((r + 1 + flat_at, 2 * r + 2)).translate((-r - 1, -r - 1))


def make_bolt(shank: float):
    """Bolt standing on its head at z=0 (shank along +z) and its winged nut."""
    _, Manifold, _ = _m()
    thread_len = NUT_T + 3.0
    head = Manifold.cylinder(HEAD_T, HEAD_R, circular_segments=48)
    body = _d_section(BOLT_R).extrude(shank).translate((0.0, 0.0, HEAD_T))
    thread = _thread(BOLT_R - BOLT_FLAT, thread_len).translate((0.0, 0.0, HEAD_T + shank))
    bolt = head + body + thread
    wings = Manifold.cube((2 * NUT_R + 10.0, 6.0, NUT_T)).translate((-(NUT_R + 5.0), -3.0, 0.0))
    nut = Manifold.cylinder(NUT_T, NUT_R, circular_segments=6) + wings
    nut = nut - _thread(BOLT_R - BOLT_FLAT, NUT_T + 2.0, grow=THREAD_CLEAR).translate((0.0, 0.0, -1.0))
    return bolt, nut


@dataclass
class RevoluteDesign:
    order: str          # "parent_low" or "parent_high"
    nut_high: bool      # nut above the stack (else below, head above)
    radius: float
    z_interface: float
    styles: tuple[str, str]
    parent_strut: object
    child_strut: object
    parent_face: int
    child_face: int


def _revolute_stack(design: RevoluteDesign, lower: float, upper: float, continuous: bool):
    """Knuckle discs with struts, stop groove and pin, plus the bolt and nut, in the joint frame.

    Returns (parent part, child part, bolt, nut, stops, (bolt, nut) posed for printing).  The child's pin runs
    in an arc groove in the parent disc that ends at the joint limits.  The
    bolt's D-shank is keyed to the disc on the nut's side, so the nut never
    turns on its thread when the joint moves; the friction is at the head.
    """
    CrossSection, Manifold, OpType = _m()
    zi, R = design.z_interface, design.radius
    low = (zi - DISC_GAP / 2 - DISC_T, zi - DISC_GAP / 2)
    high = (zi + DISC_GAP / 2, zi + DISC_GAP / 2 + DISC_T)
    parent_z, child_z = (low, high) if design.order == "parent_low" else (high, low)
    parent = _disc(R, *parent_z) + design.parent_strut
    child = _disc(R, *child_z) + design.child_strut

    # Stop pin on the child disc's inner face, into a groove in the parent disc.
    r_pin = R - 4.5
    toward_child = 1.0 if design.order == "parent_low" else -1.0
    pin_base = child_z[0] if toward_child > 0 else child_z[1]
    pin_tip = pin_base - toward_child * (DISC_GAP + STOP_DEPTH - 0.3)
    child = child + _disc(STOP_PIN_R, min(pin_base, pin_tip), max(pin_base, pin_tip), 24).translate((r_pin, 0.0, 0.0))
    groove_face = parent_z[1] if toward_child > 0 else parent_z[0]
    gz0, gz1 = sorted((groove_face, groove_face - toward_child * STOP_DEPTH))
    width = STOP_PIN_R + STOP_CLEAR
    half_angle = np.arcsin(min(1.0, width / r_pin))
    if continuous or upper - lower + 2 * half_angle >= 2 * np.pi - 0.05:
        ring = CrossSection.circle(r_pin + width, 96) - CrossSection.circle(r_pin - width, 96)
        groove = ring.extrude(gz1 - gz0 + 0.02).translate((0.0, 0.0, gz0 - 0.01))
        stops = None
    else:
        # The child at angle q carries the pin to angle q.
        segs = []
        angs = np.linspace(lower, upper, max(8, int(np.degrees(upper - lower) / 4)))
        for a0, a1 in zip(angs[:-1], angs[1:]):
            ends = [CrossSection.circle(width, 24).translate((r_pin * np.cos(a), r_pin * np.sin(a))) for a in (a0, a1)]
            segs.append(CrossSection.batch_hull(ends))
        groove = CrossSection.batch_boolean(segs, OpType.Add).extrude(gz1 - gz0 + 0.02).translate((0.0, 0.0, gz0 - 0.01))
        stops = (lower, upper)
    parent = parent - groove

    round_bore = _disc(BOLT_R + BORE_CLEAR, low[0] - 1, high[1] + 1, 48)
    keyed_bore = _d_section(BOLT_R, grow=BORE_CLEAR).extrude(high[1] - low[0] + 2).translate((0.0, 0.0, low[0] - 1))
    nut_side_is_parent = (design.nut_high == (design.order == "parent_high"))
    parent = parent - (keyed_bore if nut_side_is_parent else round_bore)
    child = child - (round_bore if nut_side_is_parent else keyed_bore)

    bolt, nut = make_bolt(shank=high[1] - low[0] + 0.5)
    printable = (bolt, nut)
    if design.nut_high:
        bolt = bolt.translate((0.0, 0.0, low[0] - HEAD_T))
        nut = nut.translate((0.0, 0.0, high[1] + 0.5))
    else:
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        bolt = _place(bolt, flip).translate((0.0, 0.0, high[1] + HEAD_T))
        nut = _place(nut, flip).translate((0.0, 0.0, low[0] - 0.5))
    return parent, child, bolt, nut, stops, printable


def _keepouts(low: tuple[float, float], high: tuple[float, float], nut_high: bool = True):
    """Room for the bolt head on one side of the stack and the nut (and a hand) on the other."""
    head_len, nut_len = HEAD_T + 2.0, NUT_T + 6.0
    if nut_high:
        return _disc(HEAD_R + 1.0, low[0] - head_len, low[0], 48) + _disc(NUT_R + 5.0, high[1], high[1] + nut_len, 48)
    return _disc(NUT_R + 5.0, low[0] - nut_len, low[0], 48) + _disc(HEAD_R + 1.0, high[1], high[1] + head_len, 48)


class Pieces:
    """A set of solids with cached bounding boxes, for quick collision tests."""

    def __init__(self, parts=()):
        self.items = []
        self._boxes = None
        for part in parts:
            self.add(part)

    def add(self, part):
        if part is None or part.is_empty():
            return self
        self.items.append((part, np.array(part.bounding_box()).reshape(2, 3)))
        self._boxes = None
        return self

    def extend(self, other: "Pieces") -> "Pieces":
        out = Pieces()
        out.items = self.items + other.items
        return out

    def hits(self, part, tol: float = 1e-3) -> bool:
        if part is None or part.is_empty() or not self.items:
            return False
        if self._boxes is None:
            self._boxes = np.array([box for _, box in self.items])
        lo, hi = np.array(part.bounding_box()).reshape(2, 3)
        near = np.all(self._boxes[:, 1] >= lo, axis=1) & np.all(self._boxes[:, 0] <= hi, axis=1)
        return any((part ^ self.items[i][0]).volume() > tol for i in np.flatnonzero(near))


def design_revolute(joint: Joint, parent: CubeSolid, child: CubeSolid, values: np.ndarray,
                    z_step: float = 2.0, parent_extra: Pieces | None = None,
                    parent_obstacle: Pieces | None = None,
                    child_obstacle: Pieces | None = None, child_slots: Pieces | None = None,
                    reasons: dict | None = None) -> RevoluteDesign | None:
    """Search knuckle placement for a revolute joint; cubes are given in the joint frame (axis z).

    Candidates vary the disc radius, which disc is on top, the bolt direction,
    the stack's position along the axis, and the strut shapes.  A candidate is
    kept when nothing collides anywhere in the joint range; the one with the
    least strut material wins.

    ``parent_extra``: parts already on the parent link (it may merge with them,
    the child must miss them).  ``parent_obstacle`` / ``child_obstacle``: the
    other neighbors' cubes swept through their own joints, fixed relative to
    the parent / child; that side's strut must miss them.  ``child_slots``:
    where the child's own joints will put their knuckles; they turn with the
    child, so the parent strut must miss them at every angle.  All in the
    joint frame (child at zero).
    """
    parent_extra = parent_extra or Pieces()
    parent_obstacle = parent_obstacle or Pieces()
    child_obstacle = child_obstacle or Pieces()
    child_slots = child_slots or Pieces()
    obstacle = parent_obstacle
    reasons = reasons if reasons is not None else {}

    def why(key):
        reasons[key] = reasons.get(key, 0) + 1

    def rot_z(a):
        return np.array([[np.cos(a), -np.sin(a), 0, 0], [np.sin(a), np.cos(a), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    parent_cube = parent.manifold()
    child_cube = child.manifold()
    child_poses = [rot_z(v) for v in values]
    child_swept = Pieces([_place(child_cube, pose) for pose in child_poses])
    avoid_stack = (Pieces([parent_cube]).extend(child_swept).extend(parent_obstacle)
                   .extend(child_obstacle).extend(parent_extra))
    for pose in child_poses:   # the moving child cube must miss the parent's existing parts
        if parent_extra.hits(_place(child_cube, pose)):
            why("child cube hits the parent's other joint parts")
            return None

    def axial(c: CubeSolid):
        z = (c.pose[:3, :3] @ (_UNIT_CORNERS * c.edge).T).T[:, 2] + c.pose[2, 3]
        return z.min(), z.max()

    (pl, ph), (cl, ch) = axial(parent), axial(child)
    zs = sorted(np.arange(min(pl, cl) - 60.0, max(ph, ch) + 60.0, z_step), key=abs)   # prefer the joint origin
    straight = [(style, 0.0) for style in ("dogleg", "dogleg_out", "direct", "fork", "dogleg_wide", "riser")]
    turned = [(style, sign * np.radians(deg)) for deg in (30, 60, 90, 120, 150, 180) for sign in (1, -1)
              for style in ("dogleg", "direct", "dogleg_wide", "riser") if not (deg == 180 and sign < 0)]
    # Struts swung around the axis only when straight ones find no layout.
    for strut_options in (straight, straight + turned):
        design = _search_revolute(parent, child, child_poses, child_swept, avoid_stack, zs, strut_options,
                                  parent_extra, obstacle, child_obstacle, child_slots, why)
        if design is not None:
            return design
    return None


def _search_revolute(parent, child, child_poses, child_swept, avoid_stack, zs, strut_options,
                     parent_extra, obstacle, child_obstacle, child_slots, why) -> RevoluteDesign | None:
    """Try knuckle placements; see :func:`design_revolute`.

    Each side's struts are checked on their own first (the child's through its
    whole motion), so a pair only needs the strut-against-strut test.
    """
    parent_cube = parent.manifold()
    parent_inner, child_inner = parent.manifold(shrink=0.3), child.manifold(shrink=0.3)
    back = [np.linalg.inv(pose) for pose in child_poses[::2]]
    best = None
    for radius in DISC_RADII:
        for order in ("parent_low", "parent_high"):
            for nut_high in (True, False):
                for zi in zs:
                    low = (zi - DISC_GAP / 2 - DISC_T, zi - DISC_GAP / 2)
                    high = (zi + DISC_GAP / 2, zi + DISC_GAP / 2 + DISC_T)
                    keep = _keepouts(low, high, nut_high)
                    if avoid_stack.hits(_disc(radius, low[0], high[1]) + keep):
                        why("no room for the knuckle on the axis")
                        continue
                    parent_z, child_z = (low, high) if order == "parent_low" else (high, low)
                    p_disc, c_disc = _disc(radius, *parent_z), _disc(radius, *child_z)
                    keep_pieces = Pieces([keep])

                    parents = []
                    for p_style, p_turn in strut_options:
                        p_strut, p_face = _strut(parent, *parent_z, radius, p_style, p_turn)
                        if p_strut.is_empty() or (p_strut ^ parent_inner).volume() > 1e-3:
                            why("parent strut cuts its own cube")
                        elif keep_pieces.hits(p_strut):
                            why("parent strut blocks the bolt or nut")
                        elif child_swept.hits(p_strut):
                            why("parent strut is in the child cube's path")
                        elif obstacle.hits(p_strut):
                            why("parent strut hits a neighboring link")
                        elif any(child_slots.hits(_place(p_strut, m)) for m in back):
                            why("parent strut sweeps through the next joint's knuckle")
                        elif Pieces([p_strut, keep]).extend(parent_extra).hits(c_disc):
                            why("parent strut cuts into the child disc")
                        else:
                            parents.append((p_style, p_turn, p_strut, p_face))
                    if not parents:
                        continue

                    static = Pieces([p_disc, keep]).extend(parent_extra)
                    children = []
                    for c_style, c_turn in strut_options:
                        c_strut, c_face = _strut(child, *child_z, radius, c_style, c_turn)
                        if c_strut.is_empty() or (c_strut ^ child_inner).volume() > 1e-3:
                            why("child strut cuts its own cube")
                            continue
                        if keep_pieces.hits(c_strut):
                            why("child strut blocks the bolt or nut")
                            continue
                        if child_obstacle.hits(c_strut):
                            why("child strut hits a neighboring link")
                            continue
                        moving = c_strut + c_disc
                        swept = [_place(c_strut, pose) for pose in child_poses]
                        if any(static.hits(part) for part in swept) or \
                                any(Pieces([parent_cube]).hits(_place(moving, pose)) for pose in child_poses):
                            why("child parts collide with the parent while turning")
                            continue
                        children.append((c_style, c_turn, c_strut, c_face, Pieces(swept)))

                    found = None
                    for p_style, p_turn, p_strut, p_face in parents:
                        for c_style, c_turn, c_strut, c_face, swept in children:
                            if swept.hits(p_strut):
                                why("child strut hits the parent strut while turning")
                                continue
                            styles = tuple(f"{st}{np.degrees(tn):+.0f}" if tn else st
                                           for st, tn in ((p_style, p_turn), (c_style, c_turn)))
                            found = RevoluteDesign(order, nut_high, radius, zi, styles,
                                                   p_strut, c_strut, p_face, c_face)
                            break
                        if found:
                            break
                    if found:
                        cost = found.parent_strut.volume() + found.child_strut.volume()
                        if best is None or cost < best[0]:
                            best = (cost, found)
                        break
        if best is not None:
            return best[1]
    return None


# ---------------------------------------------------------------------------
# Whole twin
# ---------------------------------------------------------------------------
def joint_frame(joint: Joint) -> np.ndarray:
    """Joint frame in the parent link frame, with +z along the joint axis."""
    frame = joint.origin.copy()
    frame[:3, :3] = joint.origin[:3, :3] @ _z_to(joint.axis)
    return frame


@dataclass
class TwinLink:
    name: str
    cube: LinkCube
    markers: dict[str, tuple[int, np.ndarray]]     # cube-local face -> (tag id, face raster)
    mounts: dict[str, object]                      # cube-local face -> connector part (link frame)
    module: object = None                          # mono module in the cube's frame
    dims: object = None
    legs: dict | None = None
    mount_down: dict = field(default_factory=dict)  # face -> print "down" direction (link frame)


@dataclass
class TwinBuild:
    robot: Robot
    links: dict[str, TwinLink]
    loose: list = field(default_factory=list)       # (name, link, part in that link frame, part posed to print)
    joints: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def bodies(self) -> dict[str, object]:
        """Each link's rigid body (cube plus connectors) in its link frame."""
        from aprilfold import mono

        out = {}
        for name, link in self.links.items():
            parts = [p for p in link.mounts.values()]
            cube = CubeSolid(link.cube.edge, link.cube.pose).manifold()
            out[name] = mono._union([cube, *parts])
        return out


def _mount_plate(link: TwinLink, face_name: str):
    """Blank outer layer for a mount face, with the snap legs, in the link frame."""
    from manifold3d import CrossSection
    from aprilfold import mono

    face = mono._faces()[face_name]
    half = link.dims.half
    square = CrossSection.square((2 * half + 2.0, 2 * half + 2.0), center=True)
    plate = mono._extrude(square, face, half - link.dims.tile, half) ^ mono._pyramid(face, half)
    plate = plate - mono._edge_reliefs(mono._faces(), link.dims)
    return _place(plate + mono._legs(face, link.dims, link.legs), link.cube.pose)


STACK_HALF = (HEAD_T + 2.0 + 2 * DISC_T + DISC_GAP + NUT_T + 6.0) / 2   # a knuckle stack with its head and nut room
STACK_R = NUT_R + 5.0                                                  # and its radius (the wing nut's reach)


def _knuckle_slot(joint: Joint, parent: LinkCube, child: LinkCube) -> float:
    """Where along its axis (joint frame z) a joint's knuckle stack will have room.

    The axis is blocked where either cube comes within the stack's radius;
    the slot is the free spot nearest the joint origin, or the middle of the
    widest gap when none is long enough.
    """
    inv = np.linalg.inv(joint_frame(joint))
    grid = np.array(np.meshgrid(*[np.linspace(-0.5, 0.5, 8)] * 3)).reshape(3, -1).T
    blocked = []
    for pose, edge in ((inv @ parent.pose, parent.edge), (inv @ joint.origin @ child.pose, child.edge)):
        pts = (pose[:3, :3] @ (grid * edge).T).T + pose[:3, 3]
        near = pts[np.hypot(pts[:, 0], pts[:, 1]) < STACK_R + edge / 7]
        if len(near):
            blocked.append((near[:, 2].min() - edge / 7, near[:, 2].max() + edge / 7))
    zs = sorted(np.arange(-80.0, 80.5, 1.0), key=abs)
    for z in zs:
        if all(z + STACK_HALF < lo or z - STACK_HALF > hi for lo, hi in blocked):
            return float(z)
    # No gap is long enough: the widest one between the cubes.
    return float(max(zs, key=lambda z: min([abs(z - lo) for lo, _ in blocked] + [abs(z - hi) for _, hi in blocked]
                                           if all(z < lo or z > hi for lo, hi in blocked) else [-1.0])))


def _corridor(joint: Joint, parent: LinkCube, child: LinkCube):
    """Room a joint not designed yet will need, in its parent's frame.

    The child cube swept through the joint (as pieces), and for a revolute
    joint the slot on its axis where its knuckle will go (or None).
    """
    values = sweep_values(joint)
    if joint.moves and len(values) > 2:
        span = np.degrees(values[-1] - values[0]) if joint.type != "prismatic" else values[-1] - values[0]
        values = np.linspace(values[0], values[-1], max(2, int(np.ceil(span / 20.0)) + 1))
    corners = []
    for value in values:
        pose = joint.origin @ joint_motion(joint, value) @ child.pose
        corners.append((pose[:3, :3] @ (_UNIT_CORNERS * child.edge).T).T + pose[:3, 3])
    parts = [_hull(np.vstack(pair)) for pair in zip(corners[:-1], corners[1:])] or [_hull(corners[0])]
    slot = None
    if joint.type in ("revolute", "continuous"):
        z = _knuckle_slot(joint, parent, child)
        slot = _place(_disc(STACK_R, z - STACK_HALF, z + STACK_HALF, 32), joint_frame(joint))
    return parts, slot


class JointLayoutError(ValueError):
    """No collision-free hardware layout for one joint."""

    def __init__(self, joint: str, message: str):
        super().__init__(message)
        self.joint = joint


def build_twin(robot: Robot, dict_name: str, mono_cfg, ids: list[int] | None = None,
               samples: int = 300, check_samples: int = 40, seed: int = 0) -> TwinBuild:
    """Plan cubes, design every joint, and build all printable parts of the twin.

    When a joint's knuckle finds no room, the cubes are planned again with that
    joint's reserved room kept strictly clear, a few times before giving up.
    """
    strict: set[str] = set()
    while True:
        try:
            return _build_twin(robot, dict_name, mono_cfg, ids, samples, check_samples, seed, strict)
        except JointLayoutError as exc:
            if exc.joint in strict or len(strict) >= 3:
                raise
            strict.add(exc.joint)


def _build_twin(robot: Robot, dict_name: str, mono_cfg, ids, samples, check_samples, seed,
                strict: set[str]) -> TwinBuild:
    from aprilfold import mono
    from aprilfold.generate import DICT_MAP, TagPatternGenerator, build_face_grid

    cubes = plan_cubes(robot, dict_name, mono_cfg, samples=samples, strict=strict)
    attached: dict[tuple[str, str], list] = {}
    loose, joint_records = [], {}
    fixed_children: list[tuple[Joint, str]] = []
    down: dict[tuple[str, str], np.ndarray] = {}   # print orientation of each connector (link frame)
    # Parts on each link so far (link frame); joints are designed from the root outward.
    link_parts = {name: [CubeSolid(c.edge, c.pose).manifold()] for name, c in cubes.items()}
    real = _robot_solids(robot)
    order, frontier = [], [robot.root]
    while frontier:
        link = frontier.pop(0)
        for joint in robot.child_joints(link):
            order.append(joint)
            frontier.append(joint.child)
    for joint in order:
        frame = joint_frame(joint)
        to_child = np.linalg.inv(joint.origin) @ frame
        inv = np.linalg.inv(frame)
        parent = CubeSolid(cubes[joint.parent].edge, inv @ cubes[joint.parent].pose)
        child = CubeSolid(cubes[joint.child].edge, inv @ joint.origin @ cubes[joint.child].pose)
        # Parts already on the parent link, and the other neighbors' cubes swept through
        # their own joints.  Clashes that need two joints at once (the real robot folds
        # into itself there too) are left to check_collisions.
        parent_extra = Pieces([_place(p, inv) for p in link_parts[joint.parent][1:]])
        parent_obstacle, child_obstacle, child_slots = Pieces(), Pieces(), Pieces()
        far_parent, far_child = Pieces(), Pieces()   # beyond the parent's own joint: avoided if possible
        ancestor = robot.parent_joint(joint.parent)
        while ancestor is not None:
            # Designed already: everything above each joint up the chain, swept through
            # that one joint, wherever the robot's own links are clear of this joint's.
            near = ancestor is robot.parent_joint(joint.parent)
            above = [x for x in robot.links if x not in _subtree(robot, ancestor.child) and link_parts[x]]
            for v in sweep_values(ancestor):
                frames = robot.forward({ancestor.name: v})
                to_joint = inv @ np.linalg.inv(frames[joint.parent])
                for x in above:
                    for side, pieces in ((joint.parent, parent_obstacle if near else far_parent),
                                         (joint.child, child_obstacle if near else far_child)):
                        if frozenset((x, side)) == frozenset((ancestor.parent, ancestor.child)) or \
                                _links_clear(real, frames, x, side):
                            for part in link_parts[x]:
                                pieces.add(_place(part, to_joint @ frames[x]))
            ancestor = robot.parent_joint(ancestor.parent)
        for sib in robot.child_joints(joint.parent):
            if sib is not joint:
                parts, slot = _corridor(sib, cubes[sib.parent], cubes[sib.child])
                for part in [*parts, slot]:
                    parent_obstacle.add(None if part is None else _place(part, inv))
        child_in_joint = inv @ joint.origin
        for cj in robot.child_joints(joint.child):
            parts, slot = _corridor(cj, cubes[cj.parent], cubes[cj.child])
            for part in parts:
                child_obstacle.add(_place(part, child_in_joint))
            if slot is not None:
                child_obstacle.add(_place(slot, child_in_joint))
                child_slots.add(_place(slot, child_in_joint))
        obstacle = parent_obstacle
        if joint.type in ("revolute", "continuous"):
            reasons: dict[str, int] = {}
            # Clear of the whole chain if possible; else only of the parent's own joint, and the
            # final pose check reports whatever still touches in deep folds.
            for p_obst, c_obst in ((parent_obstacle.extend(far_parent), child_obstacle.extend(far_child)),
                                   (parent_obstacle, child_obstacle)):
                design = design_revolute(joint, parent, child, sweep_values(joint), parent_extra=parent_extra,
                                         parent_obstacle=p_obst, child_obstacle=c_obst,
                                         child_slots=child_slots, reasons=reasons)
                if design is not None or not (far_parent.items or far_child.items):
                    break
            if design is None:
                top = ", ".join(f"{k} ({v})" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:3])
                raise JointLayoutError(joint.name, f"joint '{joint.name}': no collision-free knuckle layout "
                                       f"found (rejected: {top}); try a different --scale")
            p_part, c_part, bolt, nut, stops, (bolt_print, nut_print) = _revolute_stack(
                design, joint.lower, joint.upper, joint.type == "continuous")
            p_face, c_face = FACE_ORDER[design.parent_face], FACE_ORDER[design.child_face]
            loose += [(f"bolt {joint.name}", joint.parent, _place(bolt, frame), bolt_print),
                      (f"nut {joint.name}", joint.parent, _place(nut, frame), nut_print)]
            # Print each knuckle with its outer disc face on the bed.
            p_down = np.array([0, 0, -1.0 if design.order == "parent_low" else 1.0])
            down[(joint.parent, p_face)] = frame[:3, :3] @ p_down
            down[(joint.child, c_face)] = to_child[:3, :3] @ -p_down
            joint_records[joint.name] = {"hardware": "knuckle", "disc_radius_mm": design.radius,
                                         "stops": None if stops is None else list(stops)}
        elif joint.type == "prismatic":
            try:
                p_part, c_part, p_face, c_face, extra_loose, record = design_prismatic(
                    joint, parent, child, parent_extra, obstacle)
            except JointLayoutError:
                # Last resort: ignore the neighbors; the final pose check reports any clash.
                p_part, c_part, p_face, c_face, extra_loose, record = design_prismatic(
                    joint, parent, child, parent_extra, Pieces())
            loose += [(f"{n} {joint.name}", joint.parent, _place(part, frame),
                       _flat_on_bed(part, np.array([1.0, 0, 0]) if n == "pin" else np.array([0, 0, -1.0])))
                      for n, part in extra_loose]
            down[(joint.parent, p_face)] = frame[:3, :3] @ np.array([0, 0, -1.0])
            down[(joint.child, c_face)] = to_child[:3, :3] @ np.array([0, 0, -1.0])
            joint_records[joint.name] = record
        else:
            p_part, p_face, c_face, record = design_fixed(joint, parent, child, parent_extra, obstacle)
            c_part = None
            fixed_children.append((joint, c_face))
            joint_records[joint.name] = record
        joint_records[joint.name].update({"parent_face": p_face, "child_face": c_face, "type": joint.type})
        attached.setdefault((joint.parent, p_face), []).append(_place(p_part, frame))
        link_parts[joint.parent].append(_place(p_part, frame))
        if c_part is not None:
            attached.setdefault((joint.child, c_face), []).append(_place(c_part, to_child))
            link_parts[joint.child].append(_place(c_part, to_child))
        else:
            attached.setdefault((joint.child, c_face), [])

    mount_faces = {name: {f for (link, f) in attached if link == name} for name in robot.links}
    tag_gen = TagPatternGenerator(DICT_MAP[dict_name])
    n_markers = sum(6 - len(mount_faces[name]) for name in robot.links)
    if ids is None:
        ids = mono.printable_tag_ids(tag_gen, n_markers)
    elif len(ids) < n_markers:
        raise ValueError(f"need {n_markers} tag IDs, got {len(ids)}")
    else:
        split = mono.unprintable_tag_ids(tag_gen, ids[:n_markers])
        if split:
            raise ValueError(f"tag IDs {split} have black pixels that split apart; omit --ids")
    id_iter = iter(ids)

    links: dict[str, TwinLink] = {}
    for name in robot.links:
        cube = cubes[name]
        lay = cube.layout
        dims, legs = mono._check_dims(cube.edge / 2.0, lay.tag_size, lay.cell_size, lay.border_cells, mono_cfg)
        face_cells = int(round(lay.marker_pixels + 2 * lay.border_cells))
        scale = 2 if (face_cells - lay.marker_pixels) % 2 else 1
        markers = {}
        for face_name in FACE_ORDER:
            if face_name in mount_faces[name]:
                continue
            tag_id = next(id_iter)
            grid = build_face_grid([tag_gen.generate(tag_id)], 1, 1, face_cells, face_cells,
                                   lay.marker_pixels, 1, False, layout_scale=scale)
            markers[face_name] = (tag_id, grid)
        surfaces = {f: (mono._grid_section(grid, lay.cell_size / scale, dims.half), str(tid),
                        f"tag {tid} on link {name} face {f}") for f, (tid, grid) in markers.items()}
        module = mono._build_module(dims, legs, np.eye(3), surfaces, {f: str(t) for f, (t, _) in markers.items()})
        link = TwinLink(name, cube, markers, {}, module, dims, legs)
        for face_name in sorted(mount_faces[name]):
            if attached[(name, face_name)]:
                link.mounts[face_name] = mono._union([_mount_plate(link, face_name), *attached[(name, face_name)]])
        links[name] = link
    # A fixed joint's strut is one piece with both mount plates.
    for joint, c_face in fixed_children:
        p_face = joint_records[joint.name]["parent_face"]
        plate = _place(_mount_plate(links[joint.child], c_face), joint.origin)
        links[joint.parent].mounts[p_face] = links[joint.parent].mounts[p_face] + plate

    for (name, face_name), vec in down.items():
        links[name].mount_down.setdefault(face_name, vec)
    build = TwinBuild(robot, links, loose, joint_records)
    build.warnings += check_collisions(build, check_samples, seed)
    for name, cube in cubes.items():
        if cube.overshoot > 0.05:
            build.warnings.append(
                f"link '{name}': its hull only fits a {cube.fitted_edge:.1f} mm cube; using the "
                f"{cube.edge:.1f} mm minimum, which sticks out up to {cube.overshoot:.1f} mm")
    return build


def _robot_solids(robot: Robot) -> dict[str, object]:
    """Each link's own meshes as one solid (its convex hull if the meshes are not closed)."""
    from manifold3d import Manifold, Mesh
    from aprilfold import mono

    out = {}
    for name, link in robot.links.items():
        parts = []
        for mesh in link.meshes:
            solid = Manifold(Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                                  tri_verts=np.asarray(mesh.faces, np.uint32)))
            if solid.status().name != "NoError" or solid.volume() <= 0:
                solid = Manifold.hull_points(np.asarray(mesh.vertices))
            parts.append(solid)
        out[name] = mono._union(parts)
    return out


def _links_clear(real: dict, frames: dict, a: str, b: str) -> bool:
    """Whether the robot's own links ``a`` and ``b`` miss each other (to 1 mm^3) at these frames."""
    pa, pb = _place(real[a], frames[a]), _place(real[b], frames[b])
    (la, ha), (lb, hb) = [np.array(p.bounding_box()).reshape(2, 3) for p in (pa, pb)]
    if np.any(ha < lb) or np.any(hb < la):
        return True
    return (pa ^ pb).volume() <= 1.0


def check_collisions(build: TwinBuild, samples: int = 40, seed: int = 0) -> list[str]:
    """Pose the twin through its joint ranges and report links that touch.

    Random poses inside the joint limits often fold a real arm into itself, so
    a pose only counts against the twin when the robot's own meshes are clear
    there.  Neighboring links are always checked.
    """
    robot = build.robot
    bodies = build.bodies()
    real = _robot_solids(robot)
    rng = np.random.default_rng(seed)
    moving = [j for j in robot.joints if j.moves]
    configs = [{}]
    for joint in moving:
        for value in (joint.lower, joint.upper):
            configs.append({joint.name: value})
    for _ in range(samples):
        configs.append({j.name: rng.uniform(*((-np.pi, np.pi) if j.type == "continuous" else (j.lower, j.upper)))
                        for j in moving})
    adjacent = {frozenset((j.parent, j.child)) for j in robot.joints}
    rigid = {frozenset((j.parent, j.child)) for j in robot.joints if j.type == "fixed"}
    hits: dict[frozenset, tuple[float, dict]] = {}
    names = list(bodies)

    def overlap(solids, frames, a, b):
        pa, pb = _place(solids[a], frames[a]), _place(solids[b], frames[b])
        (la, ha), (lb, hb) = [np.array(p.bounding_box()).reshape(2, 3) for p in (pa, pb)]
        if np.any(ha < lb) or np.any(hb < la):
            return 0.0
        return (pa ^ pb).volume()

    for q in configs:
        frames = robot.forward(q)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if frozenset((a, b)) in rigid:
                    continue   # one printed strut joins them
                vol = overlap(bodies, frames, a, b)
                if vol <= 1e-2:
                    continue
                key = frozenset((a, b))
                if key not in adjacent and overlap(real, frames, a, b) > 1.0:
                    continue   # the robot itself collides here
                if key not in hits or vol > hits[key][0]:
                    hits[key] = (vol, q)
    warnings = []
    for key, (vol, q) in hits.items():
        a, b = sorted(key)
        pose = ", ".join(f"{k}={v:.2f}" for k, v in q.items()) or "zero pose"
        warnings.append(f"links '{a}' and '{b}' collide ({vol:.1f} mm^3) where the robot does not, at {pose}")
    return warnings


BAR = 8.0             # prismatic bar width
SLEEVE_WALL = 3.0
SLEEVE_LEN = 24.0
SLIDE_CLEAR = 0.2     # per side between bar and sleeve
STOP_BLOCK = 5.0      # fixed end of the bar (hard stop at the lower limit)
CAP_LEN = 8.0         # removable cap at the free end (hard stop at the upper limit)
PIN_R = 1.5


def _slab_toward(cube: CubeSolid, toward: np.ndarray) -> tuple[int, np.ndarray]:
    k, face_c, normal = cube.mount(toward)
    a, b = [cube.pose[:3, i] for i in range(3) if abs(cube.pose[:3, i] @ normal) < 0.5]
    side = min(0.5 * cube.edge, 3 * STRUT_W)
    return k, _box_points(face_c + normal * MOUNT_SLAB / 2, [a, b, normal], [side / 2, side / 2, MOUNT_SLAB / 2])


def design_prismatic(joint: Joint, parent: CubeSolid, child: CubeSolid, parent_extra: Pieces | None = None,
                     obstacle: Pieces | None = None):
    """Bar on the parent, sleeve on the child, along the joint axis (z in the joint frame).

    The bar is held at one end by the parent; a pinned cap closes the other
    end after the sleeve is slid on.  The fixed end and the cap stop the
    sleeve exactly at the joint limits.  Returns (parent part, child part,
    parent face, child face, loose parts, record) in the joint frame.
    """
    _, Manifold, _ = _m()
    parent_extra = parent_extra or Pieces()
    obstacle = obstacle or Pieces()
    lower, upper = joint.lower, joint.upper
    pc, cc = parent.pose[:3, 3], child.pose[:3, 3]
    parent_cube, child_cube = parent.manifold(), child.manifold()
    poses = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, v], [0, 0, 0, 1.0]]) for v in sweep_values(joint)]
    half_bar = BAR / 2
    reasons: dict[str, int] = {}

    def why(key):
        reasons[key] = reasons.get(key, 0) + 1

    best = None
    perp_p, perp_c = np.array([pc[0], pc[1]]), np.array([cc[0], cc[1]])
    across = perp_c - perp_p
    side_dir = np.array([-across[1], across[0]]) / (np.linalg.norm(across) + 1e-9)
    # Bar positions between the two cubes and to either side, nearest the middle first.
    spots = sorted(((t, sh) for t in (0.5, 0.35, 0.65, 0.2, 0.8, 0.0, 1.0)
                    for sh in (0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0)),
                   key=lambda ts: abs(ts[0] - 0.5) * 60 + abs(ts[1]))
    for t, shift in spots:
        xy = perp_p + t * across + shift * side_dir
        for root_low in (True, False):
            for s0 in sorted(np.arange(cc[2] - 40.0, cc[2] + 40.0, 4.0), key=lambda z: abs(z - cc[2])):
                sleeve_lo, sleeve_hi = s0, s0 + SLEEVE_LEN       # at q = 0, child frame along z
                if root_low:
                    bar_lo = sleeve_lo + lower - SLIDE_CLEAR - STOP_BLOCK
                    bar_hi = sleeve_hi + upper + SLIDE_CLEAR + CAP_LEN
                    root = (bar_lo, bar_lo + STOP_BLOCK)
                    cap = (bar_hi - CAP_LEN, bar_hi)
                else:
                    bar_hi = sleeve_hi + upper + SLIDE_CLEAR + STOP_BLOCK
                    bar_lo = sleeve_lo + lower - SLIDE_CLEAR - CAP_LEN
                    root = (bar_hi - STOP_BLOCK, bar_hi)
                    cap = (bar_lo, bar_lo + CAP_LEN)
                x, y = xy
                bar = Manifold.cube((BAR, BAR, bar_hi - bar_lo)).translate((x - half_bar, y - half_bar, bar_lo))
                big = BAR + 2 * SLEEVE_WALL
                root_block = Manifold.cube((big, big, STOP_BLOCK)).translate((x - big / 2, y - big / 2, root[0]))
                cap_block = Manifold.cube((big, big, CAP_LEN)).translate((x - big / 2, y - big / 2, cap[0]))
                sleeve = Manifold.cube((big, big, SLEEVE_LEN)).translate((x - big / 2, y - big / 2, sleeve_lo))
                root_c = np.array([x, y, (root[0] + root[1]) / 2])
                p_face, p_slab = _slab_toward(parent, root_c)
                root_pts = np.array([[x + sx * big / 2, y + sy * big / 2, z] for sx in (-1, 1) for sy in (-1, 1)
                                     for z in root])
                # Straight from the bar's fixed end down to the face plane, then along it,
                # so the strut stays out of the sleeve's path.
                _, face_c, normal = parent.mount(root_c)
                foot = root_pts - np.outer((root_pts - face_c) @ normal - MOUNT_SLAB, normal)
                p_strut = _hull(np.vstack([root_pts, foot])) + _hull(np.vstack([foot, p_slab]))
                c_face, c_slab = _slab_toward(child, np.array([x, y, (sleeve_lo + sleeve_hi) / 2]))
                sleeve_pts = np.array([[x + sx * big / 2, y + sy * big / 2, z] for sx in (-1, 1)
                                       for sy in (-1, 1) for z in (sleeve_lo, sleeve_hi)])
                c_strut = _hull(np.vstack([sleeve_pts, c_slab]))
                channel = Manifold.cube((BAR + 2 * SLIDE_CLEAR, BAR + 2 * SLIDE_CLEAR, 1e3)).translate(
                    (x - half_bar - SLIDE_CLEAR, y - half_bar - SLIDE_CLEAR, -500.0))
                p_part = bar + root_block + p_strut
                c_part = (sleeve + c_strut) - channel
                if (p_strut ^ parent.manifold(0.3)).volume() > 1e-3 or (c_strut ^ child.manifold(0.3)).volume() > 1e-3:
                    why("a strut cuts its own cube")
                    continue
                if obstacle.hits(p_part) or obstacle.hits(cap_block):
                    why("the bar hits a neighboring link")
                    continue
                static = Pieces([parent_cube, p_part, cap_block]).extend(parent_extra)
                if any(static.hits(_place(c_part + child_cube, pose)) for pose in poses):
                    why("the sleeve side collides with the bar side while sliding")
                    continue
                cost = p_strut.volume() + c_strut.volume() + bar.volume()
                if best is None or cost < best[0]:
                    best = (cost, p_part, c_part, p_face, c_face, cap_block, (x, y), cap)
                break
    if best is None:
        top = ", ".join(f"{k} ({v})" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:3])
        raise JointLayoutError(joint.name, f"joint '{joint.name}': no collision-free slider layout found "
                                           f"(rejected: {top}); try a different --scale")
    _, p_part, c_part, p_face, c_face, cap_block, (x, y), cap = best
    # Cross pin through cap and bar.
    zc = (cap[0] + cap[1]) / 2
    pin_hole = Manifold.cylinder(BAR + 2 * SLEEVE_WALL + 2, PIN_R + 0.15, circular_segments=24, center=True)
    pin_hole = _place(pin_hole, np.array([[0, 0, 1, x], [0, 1, 0, y], [-1, 0, 0, zc], [0, 0, 0, 1.0]]))
    socket = Manifold.cube((BAR + 2 * 0.1, BAR + 2 * 0.1, CAP_LEN)).translate((x - BAR / 2 - 0.1, y - BAR / 2 - 0.1, cap[0]))
    cap_part = cap_block - socket - pin_hole
    p_part = p_part - pin_hole
    pin = _place(Manifold.cylinder(BAR + 2 * SLEEVE_WALL, PIN_R, circular_segments=24, center=True),
                 np.array([[0, 0, 1, x], [0, 1, 0, y], [-1, 0, 0, zc], [0, 0, 0, 1.0]]))
    record = {"hardware": "slider", "stops": [lower, upper], "travel_mm": upper - lower}
    return p_part, c_part, FACE_ORDER[p_face], FACE_ORDER[c_face], [("cap", cap_part), ("pin", pin)], record


def design_fixed(joint: Joint, parent: CubeSolid, child: CubeSolid, parent_extra: Pieces | None = None,
                 obstacle: Pieces | None = None):
    """One rigid strut between facing faces of the two cubes (joint frame).

    Returns (strut, parent face, child face, record).  The child's mount plate
    is added to the same printed piece later.
    """
    obstacle = obstacle or Pieces()
    pc, cc = parent.pose[:3, 3], child.pose[:3, 3]
    p_face, p_slab = _slab_toward(parent, cc)
    c_face, c_slab = _slab_toward(child, pc)
    strut = _hull(np.vstack([p_slab, c_slab]))
    if (strut ^ parent.manifold(0.3)).volume() > 1e-3 or (strut ^ child.manifold(0.3)).volume() > 1e-3:
        raise ValueError(f"joint '{joint.name}': the fixed strut would cut through a cube")
    if obstacle.hits(strut):
        raise ValueError(f"joint '{joint.name}': the fixed strut collides with another link")
    return strut, FACE_ORDER[p_face], FACE_ORDER[c_face], {"hardware": "strut"}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
TWIN_FILES = {
    "black": "print_black.3mf",
    "white": "print_white.3mf",
    "nets": "print_nets.3mf",
    "connectors": "print_connectors.3mf",
    "hardware": "print_hardware.3mf",
    "assembly": "assembly_preview.3mf",
    "preview": "twin_preview.png",
}
COLORS = {"black": "#1A1A1A", "white": "#F2F2F2", "net": "#7FA7D9", "connector": "#E0A458", "hardware": "#C0504D"}


def link_marker_config(build: TwinBuild, name: str) -> dict:
    """Detector config for one link: marker corners in the link frame (mm)."""
    from aprilfold import generate

    link = build.links[name]
    lay, edge = link.cube.layout, link.cube.edge
    face_cells = int(round(lay.marker_pixels + 2 * lay.border_cells))
    face_defs = {fd[0]: fd for fd in generate.FACE_DEFS}
    rot, center = link.cube.pose[:3, :3], link.cube.pose[:3, 3]
    markers = []
    for face_name, (tag_id, _) in link.markers.items():
        fd = face_defs[face_name]
        corners = generate._marker_corners_on_voxel_face(
            (0, 0, 0), fd, face_cells, lay.marker_pixels, edge, lay.cell_size, (edge / 2.0,) * 3)
        normal = np.zeros(3)
        normal[fd[1]] = fd[2]
        markers.append({
            "id": int(tag_id),
            "face": face_name,
            "normal": (rot @ normal).round(9).tolist(),
            "corners_mm": (corners @ rot.T + center).round(6).tolist(),
        })
    return {
        "schema_version": 2,
        "target": {"type": "robot_link", "robot": build.robot.name, "link": name,
                   "frame": "link", "cube_edge_mm": edge},
        "dict": lay.dict_name,
        "grid": "1x1x1",
        "tag_ids": [m["id"] for m in markers],
        "markers": markers,
        "tag_size_mm": lay.tag_size,
        "cell_size_mm": lay.cell_size,
        "margin_cells": 1,
        "border_cells": lay.border_cells,
        "marker_pixels": lay.marker_pixels,
        "box_dims": [edge, edge, edge],
    }


def twin_record(build: TwinBuild, source: str, scale: float) -> dict:
    links = {}
    for name, link in build.links.items():
        links[name] = {
            "cube_edge_mm": round(link.cube.edge, 4),
            "cube_pose_mm": link.cube.pose.round(6).tolist(),
            "config": f"links/{name}/config.json",
            "markers": {face: int(tid) for face, (tid, _) in link.markers.items()},
            "mount_faces": sorted(link.mounts),
            "hull_overshoot_mm": round(link.cube.overshoot, 3),
            "merged_links": build.robot.links[name].merged,
        }
    joints = []
    for joint in build.robot.joints:
        joints.append({
            "name": joint.name, "type": joint.type, "parent": joint.parent, "child": joint.child,
            "origin_mm": joint.origin.round(6).tolist(), "axis": joint.axis.round(9).tolist(),
            "lower": joint.lower, "upper": joint.upper, **build.joints.get(joint.name, {}),
        })
    return {
        "robot": build.robot.name,
        "source": source,
        "scale": scale,
        "units": {"length": "mm", "angle": "rad", "prismatic": "mm"},
        "root": build.robot.root,
        "links": links,
        "joints": joints,
        "mimic_joints": [{"name": j.name, "type": j.type, "follows": j.mimic[0], "multiplier": j.mimic[1],
                          "offset": j.mimic[2]} for j in build.robot.mimics],
        "files": dict(TWIN_FILES),
        "warnings": build.warnings,
    }


def _flat_on_bed(part, down: np.ndarray):
    """Rotate ``down`` onto -z and set the part on the bed."""
    rot = _z_to(-np.asarray(down, dtype=float)).T
    return _place(part, np.column_stack([rot, np.zeros(3)]))


def render_twin_preview(build: TwinBuild, out_path: str) -> None:
    from aprilfold import mono
    import cv2

    def scene(q):
        frames = build.robot.forward(q)
        items = []
        for name, link in build.links.items():
            pose = frames[name] @ link.cube.pose
            module = link.module
            items += [(_place(p, pose), COLORS["net"]) for p in [*module.panels.values(), *module.skins]]
            items += [(_place(p, pose), COLORS["white"]) for p in module.white.values()]
            items += [(_place(p, pose), COLORS["black"]) for p in module.black.values()]
            items += [(_place(p, frames[name]), COLORS["connector"]) for p in link.mounts.values()]
        for _, link_name, part, _ in build.loose:
            items.append((_place(part, frames[link_name]), COLORS["hardware"]))
        return items

    moving = [j for j in build.robot.joints if j.moves]
    poses = [
        ("Zero pose", {}),
        ("Joints at 40% of upper limit", {j.name: 0.4 * (j.upper if j.type != "continuous" else np.pi) for j in moving}),
        ("Joints at 40% of lower limit", {j.name: 0.4 * (j.lower if j.type != "continuous" else -np.pi) for j in moving}),
    ]
    tiles = []
    for title, q in poses:
        img = mono._render(scene(q), 22, 40)
        cv2.putText(img, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1, cv2.LINE_AA)
        tiles.append(img)
    img = mono._render(scene({}), 70, -60)
    cv2.putText(img, "Top view", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1, cv2.LINE_AA)
    tiles.append(img)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(out_path, grid)
    print(f"Wrote {out_path} ({grid.shape[1]}x{grid.shape[0]})")


def write_twin(build: TwinBuild, out_dir: str, source: str, scale: float) -> dict:
    """Write print files, per-link detector configs, twin.json, preview and README."""
    import json
    from aprilfold import mono

    os.makedirs(out_dir, exist_ok=True)
    blacks, whites, nets, labels, net_labels = [], [], [], [], []
    for name, link in build.links.items():
        half = link.cube.edge / 2.0
        for face_name in sorted(link.markers):
            tag_id = link.markers[face_name][0]
            blacks.append(mono.print_pose(link.module.black[face_name], face_name, half))
            whites.append(mono.print_pose(link.module.white[face_name], face_name, half))
            labels.append(f"{name} {face_name} tag {tag_id}")
        nets.append(link.module.net)
        net_labels.append(f"net {name}")

    def grid_cols(n):
        return max(1, int(np.ceil(np.sqrt(n))))

    mono.write_3mf(os.path.join(out_dir, TWIN_FILES["black"]),
                   [(f"black {lb}", p, COLORS["black"]) for lb, p in zip(labels, mono.arrange(blacks, grid_cols(len(blacks))))],
                   "AprilFold twin: black lattices")
    mono.write_3mf(os.path.join(out_dir, TWIN_FILES["white"]),
                   [(f"white {lb}", p, COLORS["white"]) for lb, p in zip(labels, mono.arrange(whites, grid_cols(len(whites))))],
                   "AprilFold twin: white plates")
    mono.write_3mf(os.path.join(out_dir, TWIN_FILES["nets"]),
                   [(lb, p, COLORS["net"]) for lb, p in zip(net_labels, mono.arrange(nets, grid_cols(len(nets))))],
                   "AprilFold twin: cube nets")

    connectors, connector_labels = [], []
    for name, link in build.links.items():
        for face_name, part in link.mounts.items():
            outward = link.cube.pose[:3, :3] @ mono._faces()[face_name].normal
            connectors.append(_flat_on_bed(part, link.mount_down.get(face_name, -outward)))
            connector_labels.append(f"connector {name} {face_name}")
    if connectors:
        mono.write_3mf(os.path.join(out_dir, TWIN_FILES["connectors"]),
                       [(lb, p, COLORS["connector"]) for lb, p in
                        zip(connector_labels, mono.arrange(connectors, grid_cols(len(connectors))))],
                       "AprilFold twin: joint connectors (print with supports)")
    hardware = [(name, printable) for name, _, _, printable in build.loose]
    if hardware:
        mono.write_3mf(os.path.join(out_dir, TWIN_FILES["hardware"]),
                       [(n, p, COLORS["hardware"]) for (n, _), p in
                        zip(hardware, mono.arrange([p for _, p in hardware], grid_cols(len(hardware))))],
                       "AprilFold twin: bolts, nuts, caps and pins")

    frames = build.robot.forward({})
    assembly = []
    for name, link in build.links.items():
        pose = frames[name] @ link.cube.pose
        m = link.module
        assembly += [(f"{name} panel {f}", _place(p, pose), COLORS["net"]) for f, p in m.panels.items()]
        assembly += [(f"{name} hinge {i}", _place(p, pose), COLORS["net"]) for i, p in enumerate(m.skins, 1)]
        assembly += [(f"{name} white {f}", _place(p, pose), COLORS["white"]) for f, p in m.white.items()]
        assembly += [(f"{name} black {f}", _place(p, pose), COLORS["black"]) for f, p in m.black.items()]
        assembly += [(f"{name} connector {f}", _place(p, frames[name]), COLORS["connector"]) for f, p in link.mounts.items()]
    for label, link_name, part, _ in build.loose:
        assembly.append((label, _place(part, frames[link_name]), COLORS["hardware"]))
    floor = min(np.array(p.bounding_box())[2] for _, p, _ in assembly)
    mono.write_3mf(os.path.join(out_dir, TWIN_FILES["assembly"]),
                   [(n, p.translate((128.0, 128.0, -floor)), c) for n, p, c in assembly],
                   "AprilFold twin assembly at the zero pose (preview only, do not print)")
    render_twin_preview(build, os.path.join(out_dir, TWIN_FILES["preview"]))

    for name in build.links:
        link_dir = os.path.join(out_dir, "links", name)
        os.makedirs(link_dir, exist_ok=True)
        with open(os.path.join(link_dir, "config.json"), "w") as f:
            json.dump(link_marker_config(build, name), f, indent=2)
    record = twin_record(build, source, scale)
    with open(os.path.join(out_dir, "twin.json"), "w") as f:
        json.dump(record, f, indent=2)
    write_twin_readme(build, record, out_dir)
    return record


def write_twin_readme(build: TwinBuild, record: dict, out_dir: str) -> None:
    files = record["files"]
    link_rows = "\n".join(
        f"| `{name}` | {info['cube_edge_mm']:.1f} mm | {', '.join(f'{f} {t}' for f, t in info['markers'].items())} "
        f"| {', '.join(info['mount_faces']) or '-'} |"
        for name, info in record["links"].items()
    )
    joint_rows = "\n".join(
        f"| `{j['name']}` | {j['type']} | `{j['parent']}` -> `{j['child']}` | {j.get('hardware', '-')} | "
        + ("-" if not j.get("stops") else
           (f"{np.degrees(j['stops'][0]):.0f} to {np.degrees(j['stops'][1]):.0f} deg" if j["type"] != "prismatic"
            else f"{j['stops'][0]:.1f} to {j['stops'][1]:.1f} mm"))
        + " |"
        for j in record["joints"]
    )
    warnings = "\n".join(f"- {w}" for w in record["warnings"]) or "None."
    reduced = [f"- `{name}` also carries {', '.join(f'`{m}`' for m in info['merged_links'])} (fixed joints)."
               for name, info in record["links"].items() if info["merged_links"]]
    reduced += [f"- `{m['name']}` is not built; it follows `{m['follows']}`: "
                f"value = {m['multiplier']:.6g} * {m['follows']} + {m['offset']:.6g}." for m in record["mimic_joints"]]
    reduced_md = ("\n## Simplified\n\n" + "\n".join(reduced) + "\n") if reduced else ""
    md = f"""# Printed twin of `{record['robot']}`

![Twin preview]({files['preview']})

Every link of the robot is a marker cube, the largest that fits inside the
link's convex hull, joined to its neighbors by printed joints at the URDF joint
positions. A camera that tracks the cubes sees every link's pose, and from it
the joint angles. Scale: {record['scale']:g}.

## Links

| Link | Cube | Markers (face tag) | Joint mounts |
|------|------|--------------------|--------------|
{link_rows}

## Joints

| Joint | Type | Links | Hardware | Hard stops |
|-------|------|-------|----------|------------|
{joint_rows}
{reduced_md}
## Printing

| File | What | Color |
|------|------|-------|
| `{files['black']}` | Black lattices, one per marker | black |
| `{files['white']}` | White plates, one per marker | white |
| `{files['nets']}` | One folding net per link | any |
| `{files['connectors']}` | Joint parts: mount plate, strut, and knuckle or slider | any, with supports |
| `{files['hardware']}` | Bolts and wing nuts (and slider caps and pins) | any |

## Assembly

1. Fold each link's net into its cube (both `X` panels, then `Y`, then the
   `+Z` lid until it clicks) and fit its white plates and black lattices; each
   panel is engraved with the tag ID that goes on it.
2. Faces listed under *Joint mounts* take a connector instead of a marker:
   press the connector's legs into the panel until they click.
3. Revolute joints: stack the two knuckle discs so the stop pin sits in the
   groove, push the bolt through, and turn the wing nut until the joint holds
   its pose. The flat on the bolt keys it to the nut's disc, so moving the
   joint never loosens the nut.
4. Prismatic joints: slide the sleeve onto the bar, press the cap onto the free
   end, and push the pin through cap and bar.

## Tracking

`twin.json` describes the kinematic chain (joint origins in mm, axes, limits)
and each cube's pose in its link frame. `links/<link>/config.json` is a
standard aprilfold detector config whose marker corners are given in that
link's frame, so a detected pose is the link's pose.

## Checks

{warnings}
"""
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(md)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    import argparse
    import sys

    from aprilfold.generate import DICT_MAP, MonoConfig, parse_ids

    parser = argparse.ArgumentParser(
        prog="aprilfold twin",
        description="Build a printed twin of a robot: one marker cube per link, joined by printed joints.",
        epilog="Example:\n  %(prog)s so101.urdf.xacro -o so101_twin --last-link wrist\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("robot", help="URDF or xacro file (meshes resolved from package://, file:// or relative paths)")
    parser.add_argument("-o", "--output", default=None, help="output directory (default: <robot name>_twin)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="uniform scale of the twin (default: 1); joint angles are unchanged")
    parser.add_argument("-d", "--dict", default="4x4_250", choices=sorted(DICT_MAP),
                        help="marker dictionary (default: 4x4_250)")
    parser.add_argument("-t", "--ids", default=None, help="tag IDs: range (0-40) or comma-separated")
    parser.add_argument("--package-path", action="append", default=[], metavar="PKG=DIR",
                        help="where to find a ROS package for package:// mesh paths (repeatable)")
    parser.add_argument("--mesh-dir", action="append", default=[], metavar="DIR",
                        help="directory to search for meshes the robot file points to but that are not there "
                             "(repeatable)")
    parser.add_argument("--mono-clearance", type=float, default=None,
                        help="per-side gap between black lattice and white plate (default: 0.15)")
    parser.add_argument("--last-link", default=None, metavar="LINK",
                        help="stop the twin at this link: links beyond it (e.g. the gripper) are left out")
    parser.add_argument("--keep-fixed", action="store_true",
                        help="give links joined by fixed joints their own cubes, connected by printed struts "
                             "(default: one cube per rigid group)")
    parser.add_argument("--keep-mimic", action="store_true",
                        help="build mimic joints as independent joints (default: leave them out; they follow "
                             "the joint they mimic)")
    parser.add_argument("--quick", action="store_true", help="fewer cube orientations and collision poses")
    args = parser.parse_args(argv)

    try:
        from aprilfold import mono
        mono.require_manifold()
        import scipy  # noqa: F401
        import trimesh  # noqa: F401
    except (ImportError, RuntimeError):
        print("Error: aprilfold twin needs manifold3d, trimesh and scipy: pip install 'aprilfold[twin]'",
              file=sys.stderr)
        sys.exit(1)
    if args.scale <= 0:
        parser.error("--scale must be positive")
    packages = {}
    for item in args.package_path:
        name, sep, path = item.partition("=")
        if not sep:
            parser.error(f"--package-path expects PKG=DIR, got '{item}'")
        packages[name] = Path(path).expanduser()
    mono_cfg = MonoConfig(enabled=True)
    if args.mono_clearance is not None:
        mono_cfg.fit_clearance_mm = args.mono_clearance

    try:
        robot = load_robot(args.robot, packages, args.scale, [Path(d) for d in args.mesh_dir])
        if args.last_link:
            robot = cut_after(robot, args.last_link)
        if not args.keep_mimic:
            robot = drop_mimic(robot)
        robot = drop_empty_links(robot, merge_fixed=not args.keep_fixed)
        ids = parse_ids(args.ids, 6 * len(robot.links)) if args.ids else None
        print(f"Robot '{robot.name}': {len(robot.links)} links, {len(robot.joints)} joints, scale {args.scale:g}")
        build = build_twin(robot, args.dict, mono_cfg, ids=ids,
                           samples=60 if args.quick else 300, check_samples=10 if args.quick else 40)
        out_dir = args.output or f"{robot.name}_twin"
        record = write_twin(build, out_dir, os.path.basename(args.robot), args.scale)
    except (OSError, ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    for name, info in record["links"].items():
        print(f"  {name:16s} cube {info['cube_edge_mm']:6.1f} mm, markers {list(info['markers'].values())}")
    for warning in record["warnings"]:
        print(f"Warning: {warning}", file=sys.stderr)
    print(f"Wrote {out_dir}/ (twin.json, links/*/config.json, print files)")
