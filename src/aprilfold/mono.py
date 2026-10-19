"""Single-color build of a cube target: black, white, and net parts that snap together.

A multi-color printer lays black and white filament side by side.  Without
one, each face is split into parts that each print in one color:

* **Black lattice** - the marker's black pixels (border ring plus black data
  cells) as one connected plate, with two barbed legs under the ring.
* **White plate** - covers the whole face.  Its white pixels stand up as
  pegs and the black lattice drops into the pockets between them.
* **Net** - six panels joined by thin living hinges, printed flat and folded
  into a cube.

Every part is cut by its face's 45-degree pyramid, so faces meet exactly on
the cube's edges.

The roles cannot be swapped: white data cells sit inside the black border
ring, so a white plate with holes would fall apart into islands.  Black
pixels stay connected for most tag IDs (cells touching only at a corner do
not count); markers whose black pixels split apart are rejected.

Locking:

* The lattice's legs pass through the white plate and the net panel.  Each
  barb pushes aside a flexure beam cut into the panel and hooks behind it,
  which clamps lattice, plate, and panel together.  The beam bends within the
  panel's print plane, the strong direction for FDM parts.
* The net locks itself.  Fold the X panels up first, then the Y panels,
  whose edge tabs slide into slots inside the X panels, then the +Z lid,
  whose two tabs slide into the Y panels and snap past a flexure beam in each
  slot.  A tab can only enter a panel that is already folded, which fixes
  that order.  Once the lid is latched every panel is trapped: X by the Y
  tabs, Y by the lid tabs, and the lid by its latches.

A face with normal axis ``n`` puts its legs beside the edges it shares with
axis ``n + 2``; the net tabs sit at the middle of their edges, clear of them.

Larger targets are built from one such cube module per voxel.  Neighbors are
joined through their shared face by a spacer (both modules' outer layer) and
a rivet pushed from inside the later module into the earlier one's leg slots.
"""

from __future__ import annotations

import itertools
import os
import zipfile
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from aprilfold.generate import (
    FACE_DEFS,
    CubeConfig,
    MonoConfig,
    TagPatternGenerator,
    _camera_from_angles,
    _fmt,
)

try:
    from manifold3d import CrossSection, JoinType, Manifold, OpType
except ImportError:  # pragma: no cover - exercised only without the extra
    Manifold = None

# The net unfolds around -Z, which lies on the bed: four sides hang off it
# and +Z hangs off +X.
NET_PARENTS = {"+X": "-Z", "-X": "-Z", "+Y": "-Z", "-Y": "-Z", "+Z": "+X"}

# Snap legs under the lattice ring (mm).
LEG_THICKNESS = 2.0   # across the ring
LEG_WIDTH = 4.0       # along the ring
LEG_MARGIN = 0.3      # inset from the ring's inner edge
BARB_DEPTH = 0.6      # how far the barb sticks out of the leg
BARB_FLAT = 0.6       # straight section below the catch face
BARB_RAMP = 1.4       # lead-in taper to the leg tip
CATCH_GAP = 0.15      # axial play between the barb and the panel's inner face

# Flexure beam in the panel that the barb pushes aside (mm).
BEAM_WIDTH = 1.2
BEAM_LENGTH = 8.0
BEAM_GAP = 0.8

# Net tabs (mm): (folding panel, panel it enters, snaps shut), in fold order.
NET_TABS = [
    ("+Y", "+X", False), ("+Y", "-X", False), ("-Y", "+X", False), ("-Y", "-X", False),
    ("+Z", "+Y", True), ("+Z", "-Y", True),
]
TAB_WALL = 0.7        # panel material on each side of a tab slot
TAB_WIDTH = 6.0       # along the edge
TAB_REACH = 3.0       # how far a plain tab goes past its own panel's inner face
SWEEP_MARGIN = 0.4    # extra slot width for the arc a tab travels while folding
LATCH_BARB = 1.0      # how far a latch barb sticks out of its tab (0.4 mm past the nub)
LATCH_NUB = 1.0       # length of the beam's catch nub along the slot
LATCH_BEAM = 7.0      # flexure beam length in the receiving panel

HINGE_RADIUS = 3.0       # outer bend radius of a folded living hinge
HINGE_GAP = 0.4          # clearance between the hinge skin and the rounded panel edge
EDGE_RELIEF = 0.8        # chamfer on plate backs where a folded hinge bulges
LABEL_PIXEL = 0.8
LABEL_DEPTH = 0.4

COLORS = {"black": "#1A1A1A", "white": "#F2F2F2", "net": "#7FA7D9", "joint": "#E0A458", "rivet": "#C0504D"}

_FONT = {
    "+": [".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."],
    "-": [".....", ".....", ".....", "#####", ".....", ".....", "....."],
    "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
    "Y": ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
    "Z": ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
    "0": [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["####.", "....#", "....#", ".###.", "....#", "....#", "####."],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    "6": [".###.", "#....", "#....", "####.", "#...#", "#...#", ".###."],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    "9": [".###.", "#...#", "#...#", ".####", "....#", "....#", ".###."],
}


def require_manifold() -> None:
    if Manifold is None:
        raise RuntimeError(
            "Single-color targets need the manifold3d package. "
            "Install with `pip install 'aprilfold[mono]'`."
        )


# ---------------------------------------------------------------------------
# Tag printability
# ---------------------------------------------------------------------------
def black_is_connected(pattern: np.ndarray) -> bool:
    """True when a marker's black cells form one piece under 4-connectivity."""
    n_labels, _ = cv2.connectedComponents(pattern.astype(np.uint8), connectivity=4)
    return n_labels - 1 == 1


def printable_tag_ids(tag_gen: TagPatternGenerator, count: int) -> list[int]:
    """Return the first ``count`` IDs whose black pixels print as one lattice."""
    ids = [
        tag_id for tag_id in range(tag_gen.max_id)
        if black_is_connected(tag_gen.generate(tag_id))
    ][:count]
    if len(ids) < count:
        raise ValueError(
            f"dictionary has only {len(ids)} markers with connected black pixels; "
            f"need {count}"
        )
    return ids


def unprintable_tag_ids(tag_gen: TagPatternGenerator, tag_ids: list[int]) -> list[int]:
    return [tag_id for tag_id in tag_ids if not black_is_connected(tag_gen.generate(tag_id))]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Face:
    name: str
    axis: int
    normal: np.ndarray
    right: np.ndarray
    down: np.ndarray

    @property
    def basis(self) -> np.ndarray:
        """3x4 transform from face-local (u along right, v along down, w along normal)."""
        return np.column_stack([self.right, self.down, self.normal, np.zeros(3)])


def _unit(axis: int, sign: float = 1.0) -> np.ndarray:
    vec = np.zeros(3)
    vec[axis] = float(sign)
    return vec


def _faces() -> dict[str, _Face]:
    return {
        name: _Face(
            name=name,
            axis=normal_ax,
            normal=_unit(normal_ax, normal_sign),
            right=_unit(right_ax, right_sign),
            down=_unit(down_ax, down_sign),
        )
        for name, normal_ax, normal_sign, right_ax, right_sign, down_ax, down_sign in FACE_DEFS
    }


def _box(lo: np.ndarray, hi: np.ndarray) -> Manifold:
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return Manifold.cube(tuple(float(v) for v in hi - lo)).translate(tuple(float(v) for v in lo))


class _Frame:
    """Axis-aligned frame for building boxes and hulls directly in world space."""

    def __init__(self, origin: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.origin, self.x, self.y, self.z = origin, x, y, z

    def point(self, x: float, y: float, z: float) -> np.ndarray:
        return self.origin + x * self.x + y * self.y + z * self.z

    def box(self, xr: tuple[float, float], yr: tuple[float, float], zr: tuple[float, float]) -> Manifold:
        corners = np.array([self.point(x, y, z) for x in xr for y in yr for z in zr])
        return _box(corners.min(axis=0), corners.max(axis=0))

    def hull(self, points: list[tuple[float, float, float]]) -> Manifold:
        return Manifold.hull_points(np.array([self.point(*p) for p in points]))

    def prism(self, section: CrossSection, zr: tuple[float, float]) -> Manifold:
        """Extrude an (x, y) cross-section between z offsets in this frame."""
        basis = np.column_stack([self.x, self.y, self.z, self.origin])
        return section.extrude(zr[1] - zr[0]).translate((0.0, 0.0, zr[0])).transform(basis)


def _union(parts: list[Manifold]) -> Manifold:
    parts = [part for part in parts if not part.is_empty()]
    if not parts:
        return Manifold()
    return Manifold.batch_boolean(parts, OpType.Add)


def _extrude(section: CrossSection, face: _Face, w0: float, w1: float) -> Manifold:
    """Extrude a face-local cross-section between normal offsets ``w0`` and ``w1``."""
    return section.extrude(w1 - w0).translate((0.0, 0.0, w0)).transform(face.basis)


def _grid_section(grid: np.ndarray, cell: float, half: float) -> CrossSection:
    """Union of the True cells of a face raster, merged per row run."""
    rects = []
    rows, cols = grid.shape
    for row in range(rows):
        col = 0
        while col < cols:
            if not grid[row, col]:
                col += 1
                continue
            start = col
            while col < cols and grid[row, col]:
                col += 1
            u0 = -half + start * cell
            rect = CrossSection.square(((col - start) * cell, cell)).translate((u0, -half + row * cell))
            rects.append(rect)
    return CrossSection.batch_boolean(rects, OpType.Add)


def _label_section(text: str, pixel: float, mirror: bool) -> CrossSection:
    """Bitmap text in face-local (u, v); ``mirror=True`` reads correctly from outside.

    FACE_DEFS has right x down = +normal, so an unmirrored (u, v) drawing reads
    correctly only when viewed from inside the cube.
    """
    rows = len(_FONT["X"])
    glyphs = [np.array([[ch == "#" for ch in line] for line in _FONT[char]]) for char in text]
    grid = np.hstack([
        part for idx, glyph in enumerate(glyphs)
        for part in ([glyph] if idx == 0 else [np.zeros((rows, 1), dtype=bool), glyph])
    ])
    # Grow each pixel slightly so diagonal neighbours merge instead of pinching.
    rects = [
        CrossSection.square((pixel + 0.02, pixel + 0.02)).translate((
            (col - grid.shape[1] / 2) * pixel - 0.01,
            (row - rows / 2) * pixel - 0.01,
        ))
        for row, col in zip(*np.nonzero(grid))
    ]
    section = CrossSection.batch_boolean(rects, OpType.Add)
    return section.mirror((1.0, 0.0)) if mirror else section


def _rotation_about_line(point: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """4x4 homogeneous rotation by +90 degrees about a line."""
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot = np.round(np.eye(3) + kx + kx @ kx)  # Rodrigues at 90 degrees
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = point - rot @ point
    return mat


def _translation(offset: np.ndarray) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, 3] = offset
    return mat


def _apply(part: Manifold, mat: np.ndarray) -> Manifold:
    return part.transform(mat[:3, :])


# ---------------------------------------------------------------------------
# Part construction
# ---------------------------------------------------------------------------
@dataclass
class MonoBuild:
    """Assembled parts in target coordinates plus the flat net in bed coordinates."""
    black: dict[str, Manifold]
    white: dict[str, Manifold]
    panels: dict[str, Manifold]
    hinges: list[Manifold]   # folded hinge skins, for previews and fit checks
    net: Manifold
    dims: dict[str, float]


@dataclass(frozen=True)
class _Dims:
    half: float          # half the cube edge
    inlay: float         # black lattice thickness = white pocket depth
    base: float          # white floor under the lattice
    panel: float         # net panel thickness
    fit: float           # lattice-to-pocket clearance per side
    slot: float          # clearance around legs and tabs
    hinge: float

    @property
    def tile(self) -> float:
        return self.inlay + self.base

    @property
    def panel_outer(self) -> float:
        """Half size of the folded net's outer faces (and the hinge lines)."""
        return self.half - self.tile

    @property
    def panel_inner(self) -> float:
        return self.half - self.tile - self.panel


def _validate_dims(config: CubeConfig, mono: MonoConfig) -> tuple[_Dims, dict[str, float]]:
    sizes = set(round(v, 6) for v in config.box_dims)
    if len(sizes) != 1:
        raise ValueError("single-color cuboid targets must be 1x1x1 cubes")
    return _check_dims(config.box_dims[0] / 2.0, config.tag_size_mm, config.cell_size,
                       config.border_cells, mono)


def _check_dims(half: float, tag_size: float, cell_size: float, border_cells: float,
                mono: MonoConfig) -> tuple[_Dims, dict[str, float]]:
    """Validate one cube module of edge ``2 * half`` carrying a centered marker."""
    for label, value in [
        ("inlay_mm", mono.inlay_mm), ("base_mm", mono.base_mm), ("panel_mm", mono.panel_mm),
        ("fit_clearance_mm", mono.fit_clearance_mm), ("slot_clearance_mm", mono.slot_clearance_mm),
        ("hinge_mm", mono.hinge_mm),
    ]:
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"mono.{label} must be a non-negative number")
    if mono.inlay_mm <= 0 or mono.base_mm <= 0 or mono.hinge_mm <= 0:
        raise ValueError("mono.inlay_mm, mono.base_mm, and mono.hinge_mm must be positive")
    if mono.panel_mm < 2 * TAB_WALL + 1.2:
        raise ValueError(f"mono.panel_mm must be at least {2 * TAB_WALL + 1.2:.4g} mm to hold the edge tabs")

    dims = _Dims(
        half=half,
        inlay=mono.inlay_mm,
        base=mono.base_mm,
        panel=mono.panel_mm,
        fit=mono.fit_clearance_mm,
        slot=mono.slot_clearance_mm,
        hinge=mono.hinge_mm,
    )
    if mono.hinge_mm >= dims.panel:
        raise ValueError("mono.hinge_mm must be thinner than the net panels")

    # Legs hang under the border ring and must pass inside the folded net.
    ring_in = tag_size / 2.0 - cell_size
    leg_in = ring_in + dims.fit + LEG_MARGIN
    leg_out = leg_in + LEG_THICKNESS
    min_cell = LEG_THICKNESS + 2.0 * (dims.fit + LEG_MARGIN)
    if cell_size < min_cell:
        raise ValueError(
            f"single-color snap legs need marker cells of at least {min_cell:.3g} mm "
            f"(got {cell_size:.3g} mm); increase the tag size"
        )
    slack = dims.panel_inner - dims.slot - 0.5 - leg_out
    if slack < 0:
        need_cells = np.ceil((border_cells * cell_size - slack) / cell_size * 2.0) / 2.0
        raise ValueError(
            "single-color snap legs do not fit inside the folded net: the marker sits too "
            f"close to the cube edge. Use --border-cell {need_cells:g} or more, a larger tag, "
            "or thinner mono panel/inlay/base settings."
        )

    beam_length = min(BEAM_LENGTH, dims.panel_inner - 1.0 - LEG_WIDTH / 2.0 - dims.slot)
    if beam_length < 5.0:
        raise ValueError("single-color flexure beams do not fit; increase the tag size")

    layout = {
        "leg_in": leg_in,
        "leg_out": leg_out,
        "beam_length": beam_length,
    }
    return dims, layout


def _leg_frame(face: _Face, sign: int, dims: _Dims) -> _Frame:
    """Origin on the lattice back; x points from the face center toward a ring side."""
    return _Frame(
        origin=(dims.half - dims.inlay) * face.normal,
        x=_unit((face.axis + 2) % 3, sign),
        y=_unit((face.axis + 1) % 3),
        z=-face.normal,
    )


def _tab_frame(faces: dict[str, _Face], folding: str, receiving: str, dims: _Dims) -> _Frame:
    """Origin on the folding panel's outer face at the edge middle.

    x points toward the receiving panel (so x is depth in its thickness),
    y runs along the edge, and z points into the cube (depth along the slot).
    """
    n_fold, n_recv = faces[folding].normal, faces[receiving].normal
    return _Frame(origin=dims.panel_outer * n_fold, x=n_recv, y=np.cross(n_fold, n_recv), z=-n_fold)


def _legs(face: _Face, dims: _Dims, layout: dict[str, float]) -> Manifold:
    r0, r1 = layout["leg_in"], layout["leg_out"]
    half_w = LEG_WIDTH / 2.0
    catch = dims.base + dims.panel + CATCH_GAP
    tip = catch + BARB_FLAT + BARB_RAMP
    parts = []
    for sign in (1, -1):
        frame = _leg_frame(face, sign, dims)
        parts.append(frame.box((r0, r1), (-half_w, half_w), (-0.3, tip)))
        barb_profile = [
            (r0 + 0.3, catch), (r0 - BARB_DEPTH, catch),
            (r0 - BARB_DEPTH, catch + BARB_FLAT), (r0 + 0.3, tip),
        ]
        parts.append(frame.hull([(r, t, d) for r, d in barb_profile for t in (-half_w, half_w)]))
    return _union(parts)


def _leg_clearance(face: _Face, dims: _Dims, layout: dict[str, float]) -> Manifold:
    """Holes in the white floor wide enough for leg and barb to pass freely."""
    r0, r1, c = layout["leg_in"], layout["leg_out"], dims.slot
    half_w = LEG_WIDTH / 2.0 + c
    return _union([
        _leg_frame(face, sign, dims).box((r0 - BARB_DEPTH - c, r1 + c), (-half_w, half_w), (0.0, dims.base + 0.5))
        for sign in (1, -1)
    ])


def _flexure_cuts(face: _Face, dims: _Dims, layout: dict[str, float]) -> Manifold:
    """U-shaped cut around each flexure beam: leg slot, end gap, and back relief.

    The beam runs along the ring side, rooted on the side away from the edge
    tab that enters this panel, and its free end sits beside the leg.
    """
    r0, r1, c = layout["leg_in"], layout["leg_out"], dims.slot
    half_w = LEG_WIDTH / 2.0 + c
    root = -half_w - layout["beam_length"]
    depth = (dims.base - 0.5, dims.base + dims.panel + 0.5)
    beam_front = r0 - c
    beam_back = beam_front - BEAM_WIDTH
    rects = [
        ((beam_front, root), (r1 + c, half_w)),                                # leg slot
        ((beam_back - BEAM_GAP, root), (beam_back, half_w + BEAM_GAP)),        # back relief
        ((beam_back - BEAM_GAP, half_w), (r1 + c, half_w + BEAM_GAP)),         # end gap
    ]
    # One 2D outline: unioning coplanar 3D boxes can leave zero-width fins.
    outline = CrossSection.batch_boolean(
        [CrossSection.square((hi[0] - lo[0], hi[1] - lo[1])).translate(lo) for lo, hi in rects],
        OpType.Add,
    )
    return _union([_leg_frame(face, sign, dims).prism(outline, depth) for sign in (1, -1)])


def _latch_stations(dims: _Dims) -> dict[str, float]:
    """Depths along a latched slot, measured from the folding panel's outer face."""
    nub = dims.panel + 1.0   # free-end gap clears the inner miter edge
    catch = nub + LATCH_NUB + CATCH_GAP
    return {
        "nub": nub,
        "catch": catch,
        "tip": catch + BARB_FLAT + BARB_RAMP,
        "root": nub + LATCH_BEAM,
        "lane": TAB_WIDTH / 2.0 + LATCH_BARB + dims.slot + SWEEP_MARGIN,
    }


def _net_tab(frame: _Frame, dims: _Dims, latched: bool) -> Manifold:
    """Male tab on the folding panel, running inside the receiving panel's wall."""
    lo, hi = dims.panel_inner + TAB_WALL, dims.panel_outer - TAB_WALL
    half_w = TAB_WIDTH / 2.0
    if not latched:
        return frame.box((lo, hi), (-half_w, half_w), (0.0, dims.panel + TAB_REACH))
    st = _latch_stations(dims)
    barb = [
        (half_w - 0.3, st["catch"]), (half_w + LATCH_BARB, st["catch"]),
        (half_w + LATCH_BARB, st["catch"] + BARB_FLAT), (half_w - 0.3, st["tip"]),
    ]
    return _union([
        frame.box((lo, hi), (-half_w, half_w), (0.0, st["tip"])),
        frame.hull([(x, y, z) for y, z in barb for x in (lo, hi)]),
    ])


def _net_tab_slot(frame: _Frame, dims: _Dims, latched: bool) -> tuple[Manifold, Manifold]:
    """(cut, nub region) for the receiving panel.

    A plain slot is blind: thin walls stay on both faces so the tab also stops
    the receiving panel from swinging open.  A latched slot adds a lane for the
    barb and a flexure beam beside it, cut through the panel so the beam bends
    in the panel's print plane; a nub on the beam's free end catches the barb.
    """
    c, half_w = dims.slot, TAB_WIDTH / 2.0
    lo, hi = dims.panel_inner + TAB_WALL - c, dims.panel_outer - TAB_WALL + c
    wide = half_w + c + SWEEP_MARGIN
    depth = (dims.panel + TAB_REACH if not latched else _latch_stations(dims)["tip"]) + c
    cut = frame.box((lo, hi), (-wide, wide), (-0.5, depth))
    if not latched:
        return cut, Manifold()

    st = _latch_stations(dims)
    lane_out = st["lane"]
    beam_out = lane_out + BEAM_WIDTH
    top = st["nub"] - BEAM_GAP
    rects = [
        ((wide - 0.1, -0.5), (lane_out, st["root"])),                # barb lane
        ((beam_out, top), (beam_out + BEAM_GAP, st["root"])),        # back relief
        ((wide - 0.1, top), (beam_out + BEAM_GAP, st["nub"])),       # free-end gap
    ]
    outline = CrossSection.batch_boolean(
        [CrossSection.square((b[0] - a[0], b[1] - a[1])).translate(a) for a, b in rects],
        OpType.Add,
    )
    # The nub stays as part of the beam; cutting it out in 2D keeps the mesh seamless.
    nub = ((wide, st["nub"]), (lane_out + 0.3, st["nub"] + LATCH_NUB))
    outline = outline - CrossSection.square((nub[1][0] - nub[0][0], nub[1][1] - nub[0][1])).translate(nub[0])
    # (y along the edge, z along the slot) section, extruded through the panel.
    through = _Frame(frame.origin, frame.y, frame.z, frame.x)
    cut = cut + through.prism(outline, (dims.panel_inner - 1.0, dims.panel_outer + 1.0))
    nub_region = frame.box((dims.panel_inner - 2.0, dims.panel_outer + 2.0), (wide, lane_out),
                           (st["nub"], st["nub"] + LATCH_NUB))
    return cut, nub_region


def _pyramid(face: _Face, half: float) -> Manifold:
    """The cube's 45-degree wedge belonging to one face."""
    e = half + 1.0
    points = [np.zeros(3)] + [
        e * (su * face.right + sv * face.down + face.normal)
        for su in (-1, 1) for sv in (-1, 1)
    ]
    return Manifold.hull_points(np.array(points))


def _edge_reliefs(faces: dict[str, _Face], dims: _Dims) -> Manifold:
    """Thin diamond prisms along the 12 edges at the net's outer corners."""
    parts = []
    names = list(faces)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fa, fb = faces[a], faces[b]
            if fa.axis == fb.axis:
                continue
            corner = dims.panel_outer * (fa.normal + fb.normal)
            # Stop at the corners so the cut never reaches the third face's plate.
            along = np.cross(fa.normal, fb.normal) * dims.panel_outer
            points = [
                corner + off + end
                for off in (EDGE_RELIEF * fa.normal, -EDGE_RELIEF * fa.normal,
                            EDGE_RELIEF * fb.normal, -EDGE_RELIEF * fb.normal)
                for end in (along, -along)
            ]
            parts.append(Manifold.hull_points(np.array(points)))
    return _union(parts)


@dataclass(frozen=True)
class _Hinge:
    """A net fold: the child panel turns about an axis inset from the outer corner.

    The skin bends around a radius instead of a sharp corner, so both panels
    lose their outer corner along the fold and the flat skin is as long as the
    bent arc.
    """
    child: str
    parent: str
    basis: np.ndarray    # 3x4: x along child normal, y along parent normal, z along the fold, origin on the axis
    radius: float
    span: float          # flat skin length between the two panels
    segments: tuple[tuple[float, float], ...]   # skin extents along the fold (basis z)


def _hinges(faces: dict[str, _Face], dims: _Dims) -> list[_Hinge]:
    radius = min(HINGE_RADIUS, dims.panel)
    reach = dims.panel_outer - dims.panel
    hinges = []
    for child, parent in NET_PARENTS.items():
        n_child, n_parent = faces[child].normal, faces[parent].normal
        along = np.cross(n_child, n_parent)
        center = (dims.panel_outer - radius) * (n_child + n_parent)
        hinges.append(_Hinge(
            child=child,
            parent=parent,
            basis=np.column_stack([n_child, n_parent, along, center]),
            radius=radius,
            span=np.pi / 2.0 * (radius - dims.hinge / 2.0),
            segments=((-reach, reach),),
        ))
    return hinges


def _hinge_corner_cut(hinge: _Hinge, dims: _Dims) -> Manifold:
    """Material outside the bend radius, removed from both panels along a fold."""
    reach = hinge.radius + 2.0
    height = 2.0 * dims.panel_outer + 2.0
    quadrant = Manifold.cube((reach, reach, height)).translate((0.0, 0.0, -height / 2.0))
    inner = Manifold.cylinder(height, hinge.radius - dims.hinge - HINGE_GAP, circular_segments=48, center=True)
    return (quadrant - inner).transform(hinge.basis)


def _hinge_skin(hinge: _Hinge, dims: _Dims) -> Manifold:
    """The hinge strip as it sits once folded: quarter tubes at the outer corner."""
    parts = []
    for lo, hi in hinge.segments:
        length = hi - lo
        tube = (
            Manifold.cylinder(length, hinge.radius, circular_segments=48)
            - Manifold.cylinder(length + 1.0, hinge.radius - dims.hinge, circular_segments=48).translate((0.0, 0.0, -0.5))
        )
        quadrant = Manifold.cube((hinge.radius + 1.0, hinge.radius + 1.0, length))
        parts.append((tube ^ quadrant).translate((0.0, 0.0, lo)))
    return _union(parts).transform(hinge.basis)


def _net_transforms(faces: dict[str, _Face], dims: _Dims, hinges: list[_Hinge]) -> dict[str, np.ndarray]:
    """Unfold the net: map each folded panel onto the bed (outer faces at z=0)."""
    mats = {"-Z": _translation(np.array([0.0, 0.0, dims.panel_outer]))}
    for hinge in hinges:
        n_child = faces[hinge.child].normal
        axis_point, axis = hinge.basis[:, 3], hinge.basis[:, 2]
        mats[hinge.child] = (
            mats[hinge.parent]
            @ _translation(hinge.span * n_child)
            @ _rotation_about_line(axis_point, axis)
        )
        assert np.allclose(mats[hinge.child][:3, :3] @ n_child, [0.0, 0.0, -1.0])
    return mats


def _hinge_strips(faces: dict[str, _Face], dims: _Dims, hinges: list[_Hinge],
                  mats: dict[str, np.ndarray]) -> list[Manifold]:
    """Flat hinge skins on the bed, bridging the gap between folded-apart panels."""
    strips = []
    outer = dims.panel_outer
    for hinge in hinges:
        n_child, n_parent = faces[hinge.child].normal, faces[hinge.parent].normal
        along = hinge.basis[:, 2]
        # Where the skin leaves each panel's outer face.
        parent_end = (outer - hinge.radius) * n_child + outer * n_parent
        child_end = outer * n_child + (outer - hinge.radius) * n_parent
        for lo_z, hi_z in hinge.segments:
            corners = np.array([
                (mats[name] @ np.append(end + z * along, 1.0))[:3]
                for name, end in [(hinge.parent, parent_end), (hinge.child, child_end)]
                for z in (lo_z, hi_z)
            ])
            assert np.allclose(corners[:, 2], 0.0)
            lo, hi = corners.min(axis=0), corners.max(axis=0)
            across = int(np.argmax(np.abs(corners[2] - corners[0])[:2]))
            lo[across] -= 1.0
            hi[across] += 1.0
            lo[2], hi[2] = 0.0, dims.hinge
            strips.append(_box(lo, hi))
    return strips


@dataclass
class _Module:
    """One cube module in its own frame: local faces follow FACE_DEFS, lid on +Z."""
    black: dict[str, Manifold]    # keyed by world face name
    white: dict[str, Manifold]    # keyed by world face name
    panels: dict[str, Manifold]   # keyed by local face name
    skins: list[Manifold]
    net: Manifold                 # flat on the bed


def _label_pixel(layout: dict[str, float], text: str) -> float:
    width = 6 * len(text) - 1
    return min(LABEL_PIXEL, 2.0 * (layout["leg_in"] - 2.0) / width)


def _build_module(
    dims: _Dims,
    layout: dict[str, float],
    rot: np.ndarray,
    surfaces: dict[str, tuple[CrossSection, str, str]],
    panel_labels: dict[str, str],
) -> _Module:
    """Build a module turned by ``rot`` (local -> world, an even cube rotation).

    ``surfaces`` maps world face names that carry a marker to (black cells in
    that face's (u, v) plane, white plate label, description for errors).
    ``panel_labels`` maps world face names to text engraved on the panel.
    Everything is returned in the module's local frame.
    """
    faces = _faces()
    world_faces = {tuple(np.round(f.normal).astype(int)): f for f in faces.values()}
    to_local = np.hstack([rot.T, np.zeros((3, 1))])
    half = dims.half
    square = CrossSection.square((2 * half + 2.0, 2 * half + 2.0), center=True)
    reliefs = _edge_reliefs(faces, dims)

    def world_extrude(section: CrossSection, wface: _Face, w0: float, w1: float) -> Manifold:
        return _extrude(section, wface, w0, w1).transform(to_local)

    black: dict[str, Manifold] = {}
    white: dict[str, Manifold] = {}
    panels: dict[str, Manifold] = {}
    for name, face in faces.items():
        wface = world_faces[tuple(np.round(rot @ face.normal).astype(int))]
        pyramid = _pyramid(face, half)
        if wface.name in surfaces:
            black_cells, plate_label, desc = surfaces[wface.name]
            lattice = black_cells.offset(-dims.fit, JoinType.Miter)
            pieces = lattice.decompose()
            if len(pieces) != 1:
                raise ValueError(
                    f"{desc}: black pixels split into {len(pieces)} pieces; "
                    "choose IDs whose black cells connect edge-to-edge"
                )
            black[wface.name] = world_extrude(lattice, wface, half - dims.inlay, half) + _legs(face, dims, layout)
            pocket = world_extrude(black_cells.offset(dims.fit, JoinType.Miter), wface,
                                   half - dims.inlay, half + 1.0)
            plate = _extrude(square, face, half - dims.tile, half) ^ pyramid
            label = world_extrude(_label_section(plate_label, _label_pixel(layout, plate_label), mirror=False),
                                  wface, dims.panel_outer - 0.1, dims.panel_outer + LABEL_DEPTH)
            white[wface.name] = plate - pocket - _leg_clearance(face, dims, layout) - reliefs - label

        panel = _extrude(square, face, dims.panel_inner, dims.panel_outer) ^ pyramid
        text = panel_labels.get(wface.name)
        if text:
            panel -= world_extrude(_label_section(text, _label_pixel(layout, text), mirror=True), wface,
                                   dims.panel_outer - LABEL_DEPTH, dims.panel_outer + 0.1)
        panels[name] = panel - _flexure_cuts(face, dims, layout)

    for folding, receiving, latched in NET_TABS:
        frame = _tab_frame(faces, folding, receiving, dims)
        cut, _ = _net_tab_slot(frame, dims, latched)
        if (cut ^ _flexure_cuts(faces[receiving], dims, layout)).volume() > 0:
            raise ValueError("single-color net tabs collide with the snap legs; increase the tag size")
        panels[receiving] = panels[receiving] - cut
        panels[folding] = panels[folding] + _net_tab(frame, dims, latched)

    hinges = _hinges(faces, dims)
    for hinge in hinges:
        cut = _hinge_corner_cut(hinge, dims)
        panels[hinge.child] -= cut
        panels[hinge.parent] -= cut
    mats = _net_transforms(faces, dims, hinges)
    flat = [_apply(panels[name], mats[name]) for name in faces]
    net = _union(flat + _hinge_strips(faces, dims, hinges, mats))
    skins = [_hinge_skin(hinge, dims) for hinge in hinges]
    return _Module(black=black, white=white, panels=panels, skins=skins, net=net)


def build_mono_cube(
    config: CubeConfig,
    face_grids: dict[str, np.ndarray],
    mono: MonoConfig,
    face_tag_map: dict[str, list[int]] | None = None,
) -> MonoBuild:
    """Build every single-color part of a 1x1x1 cuboid target."""
    require_manifold()
    if config.invert:
        raise ValueError("single-color targets do not support --invert")
    dims, layout = _validate_dims(config, mono)
    cell = config.cell_size / config.layout_scale
    surfaces = {
        name: (_grid_section(grid, cell, dims.half), name,
               f"tag {face_tag_map[name]} on face {name}" if face_tag_map else f"marker on face {name}")
        for name, grid in face_grids.items()
    }
    module = _build_module(dims, layout, np.eye(3), surfaces, {name: name for name in face_grids})
    record = {
        "edge_mm": round(2 * dims.half, 4),
        "tile_mm": round(dims.tile, 4),
        "panel_outer_mm": round(2 * dims.panel_outer, 4),
        "leg_offset_mm": round(layout["leg_in"], 4),
    }
    return MonoBuild(black=module.black, white=module.white, panels=module.panels,
                     hinges=module.skins, net=module.net, dims=record)


# ---------------------------------------------------------------------------
# Multi-voxel targets: one cube module per surface voxel
# ---------------------------------------------------------------------------
RIVET_HEAD = 2.0    # rivet head thickness inside the module (mm)


def even_rotations() -> list[np.ndarray]:
    """The 12 cube rotations that permute the axes cyclically.

    A face's snap legs sit beside the edges it shares with axis ``n + 2``.
    These rotations keep that rule in world coordinates, so the leg slots of
    two neighboring modules always line up across their shared face.
    """
    rots = []
    for perm in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        for signs in itertools.product((1, -1), repeat=3):
            mat = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs)):
                mat[row, col] = sign
            if np.linalg.det(mat) > 0:
                rots.append(mat)
    return rots


def _spacer(face: _Face, dims: _Dims, layout: dict[str, float]) -> Manifold:
    """Fills both modules' outer layer at a shared face (thickness ``2 * tile``).

    ``face`` is the shared face seen from the module at the origin; the
    neighbor sits ``2 * half`` along its normal.
    """
    half, tile = dims.half, dims.tile
    points = [
        face.normal * w + size * (su * face.right + sv * face.down)
        for w, size in [(half - tile, half - tile), (half, half), (half + tile, half - tile)]
        for su in (-1, 1) for sv in (-1, 1)
    ]
    body = Manifold.hull_points(np.array(points))
    r0, r1, c = layout["leg_in"], layout["leg_out"], dims.slot
    half_w = LEG_WIDTH / 2.0 + c
    origin = (half + tile + 1.0) * face.normal
    holes = _union([
        _Frame(origin, _unit((face.axis + 2) % 3, sign), _unit((face.axis + 1) % 3), -face.normal)
        .box((r0 - BARB_DEPTH - c, r1 + c), (-half_w, half_w), (0.0, 2 * tile + 2.0))
        for sign in (1, -1)
    ])
    reliefs = _edge_reliefs(_faces(), dims)
    return body - holes - reliefs - reliefs.translate(tuple(2 * half * face.normal))


def _rivet(face: _Face, dims: _Dims, layout: dict[str, float]) -> Manifold:
    """Two barbed legs on a head bar, pushed from inside a module into its neighbor.

    The legs use the panel's snap-leg slots: they pass this module's panel,
    the spacer, and the neighbor's panel, and hook behind the neighbor's inner
    face.
    """
    r0, r1 = layout["leg_in"], layout["leg_out"]
    half_w = LEG_WIDTH / 2.0
    catch = 2 * dims.panel + 2 * dims.tile + CATCH_GAP
    tip = catch + BARB_FLAT + BARB_RAMP
    origin = dims.panel_inner * face.normal
    parts = [_Frame(origin, _unit((face.axis + 2) % 3), _unit((face.axis + 1) % 3), face.normal)
             .box((-r1, r1), (-half_w, half_w), (-RIVET_HEAD, 0.0))]
    for sign in (1, -1):
        frame = _Frame(origin, _unit((face.axis + 2) % 3, sign), _unit((face.axis + 1) % 3), face.normal)
        parts.append(frame.box((r0, r1), (-half_w, half_w), (-RIVET_HEAD + 0.1, tip)))
        barb = [(r0 + 0.3, catch), (r0 - BARB_DEPTH, catch), (r0 - BARB_DEPTH, catch + BARB_FLAT), (r0 + 0.3, tip)]
        parts.append(frame.hull([(r, t, d) for r, d in barb for t in (-half_w, half_w)]))
    return _union(parts)


def _neighbors(voxel: tuple[int, int, int]):
    for face in _faces().values():
        offset = np.round(face.normal).astype(int)
        yield face, tuple(int(v) for v in np.add(voxel, offset))


@dataclass
class MonoVoxelModule:
    voxel: tuple[int, int, int]
    center: np.ndarray
    rot: np.ndarray
    lid: str                      # world face of the module's lid
    black: dict[str, Manifold]    # world placed, keyed by world face
    white: dict[str, Manifold]
    panels: dict[str, Manifold]   # world placed, keyed by local face
    skins: list[Manifold]
    net: Manifold                 # flat on the bed


@dataclass
class MonoJoint:
    module: int                   # index of the module the rivet is pushed from
    neighbor: int                 # index of the module it snaps into
    face: str                     # world face of ``module`` toward ``neighbor``
    spacer: Manifold
    rivet: Manifold


@dataclass
class MonoVoxelBuild:
    modules: list[MonoVoxelModule]   # in assembly order
    joints: list[MonoJoint]
    skipped: list[tuple[int, int, int]]   # interior voxels left out
    dims: dict[str, float]


def build_mono_voxels(
    occupied: set[tuple[int, int, int]],
    voxel_size: float,
    tag_size: float,
    cell_size: float,
    grid_cell: float,
    markers: dict[tuple[tuple[int, int, int], str], tuple[np.ndarray, int]],
    mono: MonoConfig,
    center_abs: tuple[float, float, float],
) -> MonoVoxelBuild:
    """Build a voxel target as cube modules joined by spacers and rivets.

    ``markers`` maps (voxel, world face name) of every exposed face to its
    face raster and tag ID.  Voxels with no exposed face are left out.
    """
    require_manifold()
    half = voxel_size / 2.0
    border_cells = (voxel_size - tag_size) / 2.0 / cell_size
    dims, layout = _check_dims(half, tag_size, cell_size, border_cells, mono)
    faces = _faces()

    surface = sorted({voxel for voxel, _ in markers})
    skipped = sorted(occupied - set(surface))
    # Breadth-first order keeps every module (after the first) next to one already placed.
    order, seen = [surface[0]], {surface[0]}
    for voxel in order:
        for _, nb in _neighbors(voxel):
            if nb in occupied and nb not in seen and nb not in skipped:
                seen.add(nb)
                order.append(nb)
    if len(order) != len(surface):
        raise ValueError("single-color voxel targets need their surface voxels to connect face to face")
    index = {voxel: i for i, voxel in enumerate(order)}

    rotations = even_rotations()
    modules: list[MonoVoxelModule] = []
    for voxel in order:
        exposed = {name for (v, name) in markers if v == voxel}
        # The lid opens on an exposed face so rivets can be pushed from inside.
        rot = next(r for name in ["+Z", "-Z", "+X", "-X", "+Y", "-Y"] if name in exposed
                   for r in rotations if np.allclose(r @ [0, 0, 1], faces[name].normal))
        lid = next(n for n, f in faces.items() if np.allclose(rot @ [0, 0, 1], f.normal))
        surfaces, labels = {}, {}
        for name in exposed:
            grid, tag_id = markers[(voxel, name)]
            surfaces[name] = (_grid_section(grid, grid_cell, half), str(tag_id),
                              f"tag {tag_id} on voxel {list(voxel)} face {name}")
            labels[name] = str(tag_id)
        local = _build_module(dims, layout, rot, surfaces, labels)
        center = (np.array(voxel) + 0.5) * voxel_size - np.array(center_abs)
        place = np.column_stack([rot, center])
        modules.append(MonoVoxelModule(
            voxel=voxel, center=center, rot=rot, lid=lid,
            black={n: p.transform(place) for n, p in local.black.items()},
            white={n: p.transform(place) for n, p in local.white.items()},
            panels={n: p.transform(place) for n, p in local.panels.items()},
            skins=[p.transform(place) for p in local.skins],
            net=local.net,
        ))

    joints = []
    for i, voxel in enumerate(order):
        for face, nb in _neighbors(voxel):
            j = index.get(nb)
            if j is None or j > i:
                continue
            center = tuple(modules[i].center)
            joints.append(MonoJoint(
                module=i, neighbor=j, face=face.name,
                spacer=_spacer(face, dims, layout).translate(center),
                rivet=_rivet(face, dims, layout).translate(center),
            ))

    record = {
        "voxel_mm": round(voxel_size, 4),
        "tile_mm": round(dims.tile, 4),
        "panel_outer_mm": round(2 * dims.panel_outer, 4),
        "leg_offset_mm": round(layout["leg_in"], 4),
    }
    return MonoVoxelBuild(modules=modules, joints=joints, skipped=skipped, dims=record)


# ---------------------------------------------------------------------------
# Print layout
# ---------------------------------------------------------------------------
def print_pose(part: Manifold, face_name: str, half: float) -> Manifold:
    """Lay a tile part visible-face-down on the bed (z=0)."""
    face = _faces()[face_name]
    rot = np.eye(4)
    rot[:3, :3] = np.vstack([face.right, -face.down, -face.normal])
    rot[2, 3] = half
    return _apply(part, rot)


def arrange(parts: list[Manifold], cols: int, spacing: float = 6.0,
            center: tuple[float, float] = (128.0, 128.0)) -> list[Manifold]:
    """Grid-arrange parts on the bed, dropping each onto z=0."""
    boxes = [np.array(part.bounding_box()).reshape(2, 3) for part in parts]
    sizes = [hi - lo for lo, hi in boxes]
    col_w = max(size[0] for size in sizes) + spacing
    row_h = max(size[1] for size in sizes) + spacing
    rows = (len(parts) + cols - 1) // cols
    placed = []
    for idx, (part, (lo, hi)) in enumerate(zip(parts, boxes)):
        row, col = divmod(idx, cols)
        cx = center[0] + (col - (cols - 1) / 2.0) * col_w
        cy = center[1] - (row - (rows - 1) / 2.0) * row_h
        offset = (cx - (lo[0] + hi[0]) / 2.0, cy - (lo[1] + hi[1]) / 2.0, -lo[2])
        placed.append(part.translate(offset))
    return placed


def _mesh_arrays(part: Manifold) -> tuple[np.ndarray, np.ndarray]:
    mesh = part.to_mesh()
    return np.asarray(mesh.vert_properties)[:, :3].astype(np.float64), np.asarray(mesh.tri_verts)


def write_3mf(path: str, objects: list[tuple[str, Manifold, str]], title: str) -> None:
    """Write a plain multi-object 3MF (core spec plus display colors)."""
    colors = sorted({color for _, _, color in objects})
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<model unit="millimeter" xml:lang="en-US"'
        ' xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
        f'  <metadata name="Title">{title}</metadata>',
        '  <metadata name="Application">aprilfold</metadata>',
        "  <resources>",
        '    <basematerials id="1">',
    ]
    lines += [f'      <base name="{color}" displaycolor="{color}FF"/>' for color in colors]
    lines.append("    </basematerials>")
    for obj_id, (name, part, color) in enumerate(objects, start=2):
        verts, tris = _mesh_arrays(part)
        lines.append(
            f'    <object id="{obj_id}" name="{name}" type="model" pid="1" pindex="{colors.index(color)}">'
        )
        lines.append("      <mesh>")
        lines.append("        <vertices>")
        lines += [f'          <vertex x="{_fmt(x)}" y="{_fmt(y)}" z="{_fmt(z)}"/>' for x, y, z in verts.round(5)]
        lines.append("        </vertices>")
        lines.append("        <triangles>")
        lines += [f'          <triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in tris]
        lines.append("        </triangles>")
        lines.append("      </mesh>")
        lines.append("    </object>")
    lines.append("  </resources>")
    lines.append("  <build>")
    lines += [f'    <item objectid="{obj_id}"/>' for obj_id in range(2, len(objects) + 2)]
    lines.append("  </build>")
    lines.append("</model>")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
            '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
            '  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
            "</Types>\n",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            f'  <Relationship Target="/3D/3dmodel.model" Id="rel-{uuid4().hex[:8]}" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
            "</Relationships>\n",
        )
        zf.writestr("3D/3dmodel.model", "\n".join(lines) + "\n")
    size_kb = os.path.getsize(path) / 1024
    print(f"Wrote {path} ({size_kb:.1f} KB, {len(objects)} parts)")


# ---------------------------------------------------------------------------
# Preview rendering
# ---------------------------------------------------------------------------
def _render(items: list[tuple[Manifold, str]], elev: float, azim: float,
            size: int = 520, supersample: int = 2) -> np.ndarray:
    """Flat-shaded orthographic z-buffer render."""
    meshes = [(*_mesh_arrays(part), color) for part, color in items if not part.is_empty()]
    all_verts = np.vstack([verts for verts, _, _ in meshes])
    center = (all_verts.min(axis=0) + all_verts.max(axis=0)) / 2.0
    radius = np.linalg.norm(all_verts - center, axis=1).max()
    rvec, _, fwd = _camera_from_angles(elev, azim, 1.0)
    rot, _ = cv2.Rodrigues(rvec)
    light = rot.T @ np.array([-0.4, -0.7, 0.6])
    light /= np.linalg.norm(light)

    px = size * supersample
    scale = 0.46 * px / radius
    canvas = np.full((px, px, 3), 255, dtype=np.uint8)
    zbuf = np.full((px, px), np.inf)
    for verts, tris, color in meshes:
        cam = (verts - center) @ rot.T
        screen = cam[:, :2] * scale + px / 2.0
        tri_world = verts[tris]
        normals = np.cross(tri_world[:, 1] - tri_world[:, 0], tri_world[:, 2] - tri_world[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        keep = lengths > 1e-12
        normals[keep] /= lengths[keep, None]
        base = np.array([int(color[i:i + 2], 16) for i in (5, 3, 1)], dtype=np.float64)
        lambert = 0.45 + 0.55 * np.clip(normals @ -light, 0.0, 1.0)
        for idx in np.nonzero((normals @ fwd < 0) & keep)[0]:
            (x0, y0), (x1, y1), (x2, y2) = screen[tris[idx]]
            area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if abs(area) < 1e-9:
                continue
            xa, xb = max(int(min(x0, x1, x2)), 0), min(int(max(x0, x1, x2)) + 1, px)
            ya, yb = max(int(min(y0, y1, y2)), 0), min(int(max(y0, y1, y2)) + 1, px)
            if xa >= xb or ya >= yb:
                continue
            gx, gy = np.meshgrid(np.arange(xa, xb) + 0.5, np.arange(ya, yb) + 0.5)
            w0 = ((x1 - gx) * (y2 - gy) - (x2 - gx) * (y1 - gy)) / area
            w1 = ((x2 - gx) * (y0 - gy) - (x0 - gx) * (y2 - gy)) / area
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            depth = cam[tris[idx], 2] @ np.array([w0, w1, w2]).reshape(3, -1)
            depth = depth.reshape(gx.shape)
            region = zbuf[ya:yb, xa:xb]
            draw = inside & (depth < region)
            region[draw] = depth[draw]
            canvas[ya:yb, xa:xb][draw] = base * lambert[idx]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def render_mono_preview(build: MonoBuild, out_path: str) -> None:
    """Write a 2x2 preview: exploded views, the assembled cube, and the flat net."""
    faces = _faces()
    edge = build.dims["edge_mm"]
    folded_net = [(part, COLORS["net"]) for part in [*build.panels.values(), *build.hinges]]
    exploded = list(folded_net)
    for name, face in faces.items():
        exploded.append((build.white[name].translate(tuple(0.35 * edge * face.normal)), COLORS["white"]))
        exploded.append((build.black[name].translate(tuple(0.7 * edge * face.normal)), COLORS["black"]))
    assembled = folded_net + [
        (part, COLORS[kind]) for kind, group in [("white", build.white), ("black", build.black)]
        for part in group.values()
    ]
    views = [
        ("Exploded (top)", _render(exploded, 28, 35)),
        ("Exploded (bottom)", _render(exploded, -28, 215)),
        ("Assembled", _render(assembled, 25, 35)),
        ("Net, printed flat", _render([(build.net, COLORS["net"])], 60, -90)),
    ]
    tiles = []
    for title, image in views:
        cv2.putText(image, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1, cv2.LINE_AA)
        tiles.append(image)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(out_path, grid)
    print(f"Wrote {out_path} ({grid.shape[1]}x{grid.shape[0]})")


def render_voxel_preview(build: MonoVoxelBuild, out_path: str) -> None:
    """Write a 2x2 preview: assembled, exploded modules, joint detail, one flat net."""
    faces = _faces()
    voxel = build.dims["voxel_mm"]
    centers = np.array([m.center for m in build.modules])
    middle = (centers.min(axis=0) + centers.max(axis=0)) / 2.0

    def module_items(m: MonoVoxelModule, shift: np.ndarray, plates: float) -> list:
        items = [(p.translate(tuple(shift)), COLORS["net"]) for p in [*m.panels.values(), *m.skins]]
        for name in m.white:
            out = tuple(shift + plates * voxel * faces[name].normal)
            items.append((m.white[name].translate(out), COLORS["white"]))
            items.append((m.black[name].translate(tuple(np.add(out, 0.3 * plates * voxel * faces[name].normal))),
                          COLORS["black"]))
        return items

    assembled, exploded = [], []
    for m in build.modules:
        assembled += module_items(m, np.zeros(3), 0.0)
        exploded += module_items(m, 0.6 * (m.center - middle), 0.0)
    assembled += [(j.spacer, COLORS["net"]) for j in build.joints]
    for j in build.joints:
        a, b = build.modules[j.module], build.modules[j.neighbor]
        mid = 0.6 * ((a.center + b.center) / 2.0 - middle)
        exploded.append((j.spacer.translate(tuple(mid)), COLORS["joint"]))
    first = build.joints[0] if build.joints else None
    detail = []
    if first is not None:
        a, b = build.modules[first.module], build.modules[first.neighbor]
        gap = 0.35 * voxel * faces[first.face].normal
        # Pull the two modules apart along the joint; the rivet stays with its module.
        detail = module_items(a, -gap, 0.0) + module_items(b, gap, 0.0)
        detail += [(first.spacer, COLORS["joint"]), (first.rivet.translate(tuple(-gap)), COLORS["rivet"])]
    joint_azim = {0: 0.0, 1: 90.0, 2: 0.0}[faces[first.face].axis] if first is not None else 35.0
    views = [
        ("Assembled", _render(assembled, 25, 35)),
        ("Modules apart", _render(exploded, 28, 35)),
        ("Joint: spacer + rivet", _render(detail, 15, joint_azim + 25) if detail else _render(assembled, 25, 35)),
        ("One module's net, printed flat", _render([(build.modules[0].net, COLORS["net"])], 60, -90)),
    ]
    tiles = []
    for title, image in views:
        cv2.putText(image, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1, cv2.LINE_AA)
        tiles.append(image)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(out_path, grid)
    print(f"Wrote {out_path} ({grid.shape[1]}x{grid.shape[0]})")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
MONO_FILES = {
    "black": "print_black.3mf",
    "white": "print_white.3mf",
    "net": "print_net.3mf",
    "assembly": "assembly_preview.3mf",
    "preview": "mono_preview.png",
}


def write_mono_target(
    config: CubeConfig,
    face_grids: dict[str, np.ndarray],
    face_tag_map: dict[str, list[int]],
    mono: MonoConfig,
    out_dir: str,
) -> dict[str, Any]:
    """Build and write all single-color print files; return the config record."""
    build = build_mono_cube(config, face_grids, mono, face_tag_map)
    half = build.dims["edge_mm"] / 2.0
    names = list(build.black)

    blacks = arrange([print_pose(build.black[n], n, half) for n in names], cols=3)
    whites = arrange([print_pose(build.white[n], n, half) for n in names], cols=3)
    net = arrange([build.net], cols=1)
    write_3mf(os.path.join(out_dir, MONO_FILES["black"]),
              [(f"black {n} (tag {face_tag_map[n][0]})", part, COLORS["black"]) for n, part in zip(names, blacks)],
              "AprilFold single-color black lattices")
    write_3mf(os.path.join(out_dir, MONO_FILES["white"]),
              [(f"white {n} (tag {face_tag_map[n][0]})", part, COLORS["white"]) for n, part in zip(names, whites)],
              "AprilFold single-color white plates")
    write_3mf(os.path.join(out_dir, MONO_FILES["net"]), [("net", net[0], COLORS["net"])],
              "AprilFold single-color folding net")

    lift = (128.0, 128.0, half)
    assembly = [
        (f"{kind} {n}", group[n].translate(lift), COLORS[kind])
        for kind, group in [("net", build.panels), ("white", build.white), ("black", build.black)]
        for n in names
    ] + [(f"net hinge {i}", skin.translate(lift), COLORS["net"]) for i, skin in enumerate(build.hinges, 1)]
    write_3mf(os.path.join(out_dir, MONO_FILES["assembly"]), assembly,
              "AprilFold single-color assembly (preview only, do not print)")
    render_mono_preview(build, os.path.join(out_dir, MONO_FILES["preview"]))

    return {
        "files": dict(MONO_FILES),
        "inlay_mm": mono.inlay_mm,
        "base_mm": mono.base_mm,
        "panel_mm": mono.panel_mm,
        "fit_clearance_mm": mono.fit_clearance_mm,
        "slot_clearance_mm": mono.slot_clearance_mm,
        "hinge_mm": mono.hinge_mm,
        "net_parents": dict(NET_PARENTS),
        **build.dims,
    }


MONO_VOXEL_FILES = {
    "black": "print_black.3mf",
    "white": "print_white.3mf",
    "net": "print_nets.3mf",
    "joints": "print_joints.3mf",
    "assembly": "assembly_preview.3mf",
    "preview": "mono_preview.png",
}


def _pose(part: Manifold, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Manifold:
    """Rotate so the given world directions become the bed's x, y, z."""
    return part.transform(np.column_stack([np.vstack([x, y, z]), np.zeros(3)]))


def write_mono_voxel_target(
    occupied: set[tuple[int, int, int]],
    voxel_size: float,
    tag_size: float,
    cell_size: float,
    grid_cell: float,
    markers: dict[tuple[tuple[int, int, int], str], tuple[np.ndarray, int]],
    mono: MonoConfig,
    center_abs: tuple[float, float, float],
    out_dir: str,
) -> dict[str, Any]:
    """Build and write a multi-voxel single-color target; return the config record."""
    build = build_mono_voxels(occupied, voxel_size, tag_size, cell_size, grid_cell, markers, mono, center_abs)
    faces = _faces()
    half = voxel_size / 2.0
    tags = {key: tag_id for key, (_, tag_id) in markers.items()}

    blacks, whites, labels = [], [], []
    for m in build.modules:
        for name in sorted(m.black):
            blacks.append(print_pose(m.black[name].translate(tuple(-m.center)), name, half))
            whites.append(print_pose(m.white[name].translate(tuple(-m.center)), name, half))
            labels.append(f"tag {tags[(m.voxel, name)]}")
    cols = int(np.ceil(np.sqrt(len(blacks))))
    write_3mf(os.path.join(out_dir, MONO_VOXEL_FILES["black"]),
              [(f"black {lb}", p, COLORS["black"]) for lb, p in zip(labels, arrange(blacks, cols))],
              "AprilFold single-color black lattices")
    write_3mf(os.path.join(out_dir, MONO_VOXEL_FILES["white"]),
              [(f"white {lb}", p, COLORS["white"]) for lb, p in zip(labels, arrange(whites, cols))],
              "AprilFold single-color white plates")
    nets = arrange([m.net for m in build.modules], int(np.ceil(np.sqrt(len(build.modules)))))
    write_3mf(os.path.join(out_dir, MONO_VOXEL_FILES["net"]),
              [(f"net module {k}", p, COLORS["net"]) for k, p in enumerate(nets, 1)],
              "AprilFold single-color module nets")

    joint_parts = []
    for j in build.joints:
        f = faces[j.face]
        center = build.modules[j.module].center
        joint_parts.append((f"spacer {j.module + 1}-{j.neighbor + 1}",
                            _pose(j.spacer.translate(tuple(-center - half * f.normal)), f.right, f.down, f.normal),
                            COLORS["joint"]))
        # Rivets print on their side so the legs run along the layers.
        joint_parts.append((f"rivet {j.module + 1}-{j.neighbor + 1}",
                            _pose(j.rivet.translate(tuple(-center)), _unit((f.axis + 2) % 3), f.normal,
                                  _unit((f.axis + 1) % 3, f.normal.sum())),
                            COLORS["rivet"]))
    if joint_parts:
        placed = arrange([p for _, p, _ in joint_parts], int(np.ceil(np.sqrt(len(joint_parts)))))
        write_3mf(os.path.join(out_dir, MONO_VOXEL_FILES["joints"]),
                  [(n, p, c) for (n, _, c), p in zip(joint_parts, placed)],
                  "AprilFold single-color spacers and rivets")

    lift = np.array([128.0, 128.0, 0.0])
    assembly = []
    for k, m in enumerate(build.modules, 1):
        assembly += [(f"module {k} panel {n}", p, COLORS["net"]) for n, p in m.panels.items()]
        assembly += [(f"module {k} hinge {i}", p, COLORS["net"]) for i, p in enumerate(m.skins, 1)]
        assembly += [(f"module {k} white {n}", p, COLORS["white"]) for n, p in m.white.items()]
        assembly += [(f"module {k} black {n}", p, COLORS["black"]) for n, p in m.black.items()]
    for j in build.joints:
        assembly += [(f"spacer {j.module + 1}-{j.neighbor + 1}", j.spacer, COLORS["joint"]),
                     (f"rivet {j.module + 1}-{j.neighbor + 1}", j.rivet, COLORS["rivet"])]
    floor = min(np.array(p.bounding_box())[2] for _, p, _ in assembly)
    write_3mf(os.path.join(out_dir, MONO_VOXEL_FILES["assembly"]),
              [(n, p.translate(tuple(lift - [0, 0, floor])), c) for n, p, c in assembly],
              "AprilFold single-color assembly (preview only, do not print)")
    render_voxel_preview(build, os.path.join(out_dir, MONO_VOXEL_FILES["preview"]))

    modules = []
    for k, m in enumerate(build.modules, 1):
        modules.append({
            "module": k,
            "voxel": list(m.voxel),
            "lid": m.lid,
            "tags": {name: tags[(m.voxel, name)] for name in sorted(m.black)},
            "rivets_into": [
                {"module": j.neighbor + 1, "face": j.face}
                for j in build.joints if j.module == k - 1
            ],
        })
    return {
        "files": dict(MONO_VOXEL_FILES) if build.joints else {
            key: value for key, value in MONO_VOXEL_FILES.items() if key != "joints"
        },
        "inlay_mm": mono.inlay_mm,
        "base_mm": mono.base_mm,
        "panel_mm": mono.panel_mm,
        "fit_clearance_mm": mono.fit_clearance_mm,
        "slot_clearance_mm": mono.slot_clearance_mm,
        "hinge_mm": mono.hinge_mm,
        "net_parents": dict(NET_PARENTS),
        "modules": modules,
        "joints": len(build.joints),
        "skipped_voxels": [list(v) for v in build.skipped],
        **build.dims,
    }
