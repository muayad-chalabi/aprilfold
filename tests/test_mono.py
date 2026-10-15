import itertools
import json
import sys
import textwrap
import zipfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

pytest.importorskip("manifold3d")

from aprilcube import generate, mono  # noqa: E402
from aprilcube.detect import load_cube_config  # noqa: E402


NS = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}


def _cube(tag_size=30.0, border=1.0, dict_name="4x4_50"):
    config = generate.CubeConfig(
        1, 1, 1, generate.DICT_MAP[dict_name], dict_name, [], tag_size, border_cells=border,
    )
    config.compute()
    tag_gen = generate.TagPatternGenerator(config.dict_id)
    config.tag_ids = mono.printable_tag_ids(tag_gen, 6)
    grids, tag_map = {}, {}
    for tag_id, face_def in zip(config.tag_ids, generate.FACE_DEFS):
        rows, cols, down, right = config.face_layout(face_def)
        grids[face_def[0]] = generate.build_face_grid(
            [tag_gen.generate(tag_id)], rows, cols, down, right,
            config.marker_pixels, config.margin_cells, False, layout_scale=config.layout_scale,
        )
        tag_map[face_def[0]] = [tag_id]
    return config, grids, tag_map


@pytest.fixture(scope="module")
def built():
    config, grids, tag_map = _cube()
    build = mono.build_mono_cube(config, grids, generate.MonoConfig(enabled=True), tag_map)
    return config, tag_map, build


def _run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["aprilcube generate", *args])
    generate.main()


def test_printable_ids_skip_markers_with_split_black_pixels():
    tag_gen = generate.TagPatternGenerator(generate.DICT_MAP["4x4_50"])

    assert not mono.black_is_connected(tag_gen.generate(0))
    assert mono.printable_tag_ids(tag_gen, 6) == [1, 2, 3, 4, 5, 6]
    assert mono.unprintable_tag_ids(tag_gen, [0, 1]) == [0]


def test_parts_are_single_solids_and_do_not_collide(built):
    _, _, build = built
    parts = {
        f"{kind}{name}": part
        for kind, group in [("black", build.black), ("white", build.white), ("panel", build.panels)]
        for name, part in group.items()
    }
    for name, part in parts.items():
        assert len(part.decompose()) == 1, name
    assert len(build.net.decompose()) == 1

    parts.update({f"hinge{i}": skin for i, skin in enumerate(build.hinges)})
    for (name_a, a), (name_b, b) in itertools.combinations(parts.items(), 2):
        if name_a.startswith("panel") and name_b.startswith("hinge"):
            continue  # hinge skins are part of the net
        assert (a ^ b).volume() < 1e-4, (name_a, name_b)


def _hinges(config):
    dims, _ = mono._validate_dims(config, generate.MonoConfig(enabled=True))
    return {hinge.child: hinge for hinge in mono._hinges(mono._faces(), dims)}


def _swing(part, hinge, normal, degrees):
    """Turn a part with its folded panel about the panel's hinge, opening it by ``degrees``."""
    axis, center = hinge.basis[:, 2], hinge.basis[:, 3]
    for k in (axis, -axis):
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        theta = np.radians(degrees)
        rot = np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * kx @ kx
        offset = center - rot @ center
        if (rot @ normal + offset) @ normal > 1.0:
            return part.transform(np.column_stack([rot, offset]))
    raise AssertionError("no opening direction")


def test_barbs_hook_the_panel(built):
    _, _, build = built
    for name, face in mono._faces().items():
        pulled = build.black[name].translate(tuple(0.3 * face.normal))
        assert (pulled ^ build.panels[name]).volume() > 0.05, f"{name} lattice pulls free"


def test_net_locks_itself(built):
    config, _, build = built
    faces = mono._faces()
    for name in ["+X", "-X", "+Y", "-Y"]:
        lifted = build.panels[name].translate(tuple(0.5 * faces[name].normal))
        assert any(
            (lifted ^ build.panels[other]).volume() > 1e-3 for other in faces if other != name
        ), f"{name} panel can swing open"

    lid = _swing(build.panels["+Z"], _hinges(config)["+Z"], faces["+Z"].normal, 1.0)
    for side in ["+Y", "-Y"]:
        assert (lid ^ build.panels[side]).volume() > 0.03, f"lid latch on {side} does not catch"


def test_tabs_clear_their_slots_while_folding(built):
    config, _, build = built
    dims, _ = mono._validate_dims(config, generate.MonoConfig(enabled=True))
    faces, hinges = mono._faces(), _hinges(config)
    for folding, receiving, latched in mono.NET_TABS:
        frame = mono._tab_frame(faces, folding, receiving, dims)
        tab = mono._net_tab(frame, dims, latched)
        _, nub = mono._net_tab_slot(frame, dims, latched)
        slot_panel = build.panels[receiving] - nub   # the nub is meant to flex aside
        for degrees in (0.5, 2, 5, 10, 20, 30):
            moved = _swing(tab, hinges[folding], faces[folding].normal, degrees)
            assert (moved ^ slot_panel).volume() < 1e-4, (folding, receiving, degrees)


def test_assembled_faces_decode_as_their_markers(built):
    config, tag_map, build = built
    half = config.box_dims[0] / 2.0
    ppm = 20
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(config.dict_id))
    for name, face in mono._faces().items():
        to_face = np.vstack([face.right, face.down, face.normal])
        to_face = np.hstack([to_face, np.zeros((3, 1))])
        section = build.black[name].transform(to_face).slice(half - 0.05)
        image = np.full((int(2 * half * ppm) + 40,) * 2, 255, np.uint8)
        polys = [np.round((p + half) * ppm + 20).astype(np.int32) for p in section.to_polygons()]
        cv2.fillPoly(image, polys, 0)
        # Right x down = +normal in FACE_DEFS, so the outside view is mirrored.
        _, ids, _ = detector.detectMarkers(cv2.flip(image, 1))

        assert ids is not None and ids.ravel().tolist() == tag_map[name]


def test_cli_writes_single_color_print_files(tmp_path, monkeypatch):
    out_dir = tmp_path / "mono_cube"
    _run_cli(monkeypatch, "--grid", "1x1x1", "--dict", "4x4_50", "--tag-size", "30",
             "--mono", "-o", str(out_dir))

    config = json.loads((out_dir / "config.json").read_text())
    assert config["tag_ids"] == [1, 2, 3, 4, 5, 6]
    assert config["mono"]["files"]["black"] == "print_black.3mf"
    assert not (out_dir / "cube.3mf").exists()
    assert "--mono" in (out_dir / "README.md").read_text()
    load_cube_config(str(out_dir / "config.json"))

    expected_objects = {"print_black.3mf": 6, "print_white.3mf": 6, "print_net.3mf": 1}
    for filename, count in expected_objects.items():
        with zipfile.ZipFile(out_dir / filename) as archive:
            root = ET.fromstring(archive.read("3D/3dmodel.model"))
        objects = root.findall(".//m:object", NS)
        assert len(objects) == count
        for obj in objects:
            zs = [float(v.attrib["z"]) for v in obj.findall(".//m:vertex", NS)]
            assert min(zs) == pytest.approx(0.0, abs=1e-4)


def test_cli_rejects_unprintable_ids(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, "--mono", "--ids", "0-5", "-o", str(tmp_path / "split"))
    assert "--ids 1,2,3,4,5,6" in capsys.readouterr().err


RING = [(x, y, 0) for x in range(3) for y in range(3) if (x, y) != (1, 1)]


def _voxel_markers(occupied, dict_name="apriltag_36h11", tag=30.0, border=1.5):
    tag_gen = generate.TagPatternGenerator(generate.DICT_MAP[dict_name])
    mp = tag_gen.marker_pixels
    face_cells = int(round(mp + 2 * border))
    scale = 2 if (face_cells - mp) % 2 else 1
    exposed = [
        (v, fd[0]) for v in sorted(occupied) for fd in generate.FACE_DEFS
        if tuple(np.add(v, np.eye(3, dtype=int)[fd[1]] * fd[2])) not in occupied
    ]
    ids = mono.printable_tag_ids(tag_gen, len(exposed))
    markers = {
        key: (generate.build_face_grid([tag_gen.generate(tid)], 1, 1, face_cells, face_cells,
                                       mp, 1, False, layout_scale=scale), tid)
        for key, tid in zip(exposed, ids)
    }
    cell = tag / mp
    voxel = face_cells * cell
    lo, hi = np.min(list(occupied), axis=0), np.max(list(occupied), axis=0)
    center_abs = tuple((lo + hi + 1) * voxel / 2.0)
    build = mono.build_mono_voxels(set(occupied), voxel, tag, cell, cell / scale, markers,
                                   generate.MonoConfig(enabled=True), center_abs)
    return build, markers, voxel, tag_gen


@pytest.fixture(scope="module")
def ring():
    return _voxel_markers(set(RING))


def test_ring_modules_do_not_collide_and_rivets_hook(ring):
    build, markers, _, _ = ring
    assert len(build.modules) == 8 and len(build.joints) == 8   # the loop closes
    parts = []
    for i, m in enumerate(build.modules):
        parts += [((i, "panel"), p) for p in m.panels.values()]
        parts += [((i, "plate"), p) for p in [*m.black.values(), *m.white.values()]]
    parts += [((None, "joint"), p) for j in build.joints for p in (j.spacer, j.rivet)]
    for (_, part) in parts:
        assert len(part.decompose()) == 1
    boxes = [np.array(p.bounding_box()).reshape(2, 3) for _, p in parts]
    for i, j in itertools.combinations(range(len(parts)), 2):
        (lo_a, hi_a), (lo_b, hi_b) = boxes[i], boxes[j]
        if np.any(hi_a < lo_b) or np.any(hi_b < lo_a):
            continue
        assert (parts[i][1] ^ parts[j][1]).volume() < 1e-4, (parts[i][0], parts[j][0])

    faces = mono._faces()
    for joint in build.joints:
        pulled = joint.rivet.translate(tuple(-0.3 * faces[joint.face].normal))
        neighbor = build.modules[joint.neighbor]
        assert sum((pulled ^ p).volume() for p in neighbor.panels.values()) > 0.05


def test_ring_rivets_only_push_through_their_two_panels(ring):
    build, _, _, _ = ring
    faces = mono._faces()
    joint = build.joints[-1]   # the loop-closing joint
    others = [
        p for i, m in enumerate(build.modules) for p in [*m.black.values(), *m.white.values(),
                                                         *(m.panels.values() if i not in (joint.module, joint.neighbor) else [])]
    ] + [j.spacer for j in build.joints]
    for depth in np.arange(0.5, 9.6, 1.0):
        moved = joint.rivet.translate(tuple(-depth * faces[joint.face].normal))
        assert all((moved ^ p).volume() < 1e-4 for p in others), depth


def test_ring_faces_decode(ring):
    build, markers, voxel, tag_gen = ring
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(generate.DICT_MAP["apriltag_36h11"]))
    half, ppm = voxel / 2.0, 10
    for m in build.modules:
        for name, lattice in m.black.items():
            face = mono._faces()[name]
            to_face = np.hstack([np.vstack([face.right, face.down, face.normal]), np.zeros((3, 1))])
            section = lattice.translate(tuple(-m.center)).transform(to_face).slice(half - 0.05)
            image = np.full((int(2 * half * ppm) + 40,) * 2, 255, np.uint8)
            cv2.fillPoly(image, [np.round((p + half) * ppm + 20).astype(np.int32) for p in section.to_polygons()], 0)
            _, ids, _ = detector.detectMarkers(cv2.flip(image, 1))
            assert ids is not None and ids.ravel().tolist() == [markers[(m.voxel, name)][1]]


def test_cli_mono_grid_builds_joined_modules(tmp_path, monkeypatch):
    out_dir = tmp_path / "grid"
    _run_cli(monkeypatch, "--grid", "2x1x1", "--dict", "apriltag_36h11", "--tag-size", "30",
             "--border-cell", "1.5", "--mono", "-o", str(out_dir))
    config = json.loads((out_dir / "config.json").read_text())
    assert config["target"]["type"] == "voxel_cuboids"
    assert len(config["markers"]) == 10 and config["mono"]["joints"] == 1
    assert config["mono"]["modules"][1]["rivets_into"] == [{"module": 1, "face": "-X"}]
    assert not (out_dir / "cube.3mf").exists()
    for name in ["print_black.3mf", "print_white.3mf", "print_nets.3mf", "print_joints.3mf"]:
        assert (out_dir / name).exists()
    load_cube_config(str(out_dir / "config.json"))


def test_marker_too_close_to_edge_suggests_a_border():
    config, grids, tag_map = _cube(tag_size=24.0, border=1.0)
    with pytest.raises(ValueError, match="--border-cell 1.5"):
        mono.build_mono_cube(config, grids, generate.MonoConfig(enabled=True), tag_map)

    config, grids, tag_map = _cube(tag_size=24.0, border=1.5)
    mono.build_mono_cube(config, grids, generate.MonoConfig(enabled=True), tag_map)


def test_yaml_mono_settings(tmp_path):
    spec_path = tmp_path / "mono.yaml"
    spec_path.write_text(textwrap.dedent("""
        shape: {type: cuboid, grid: 1x1x1}
        mono:
          fit_clearance_mm: 0.2
          panel_mm: 3.6
    """))

    spec = generate.load_generation_spec(spec_path)

    assert spec.mono.enabled
    assert spec.mono.fit_clearance_mm == 0.2
    assert spec.mono.panel_mm == 3.6

    spec_path.write_text("mono: {clearance: 0.2}\n")
    with pytest.raises(ValueError, match="Unknown mono setting"):
        generate.load_generation_spec(spec_path)
