import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("manifold3d")
pytest.importorskip("trimesh")
pytest.importorskip("scipy")

from aprilcube import generate, twin  # noqa: E402
from aprilcube.detect import load_cube_config  # noqa: E402

DEMO = Path(__file__).parent / "data" / "twin_demo.urdf"


@pytest.fixture(scope="module")
def demo():
    robot = twin.drop_empty_links(twin.load_robot(DEMO))
    build = twin.build_twin(robot, "4x4_250", generate.MonoConfig(enabled=True), samples=40, check_samples=10)
    return robot, build


def _joint(robot, name):
    return next(j for j in robot.joints if j.name == name)


def _overlap(a, b):
    return (a ^ b).volume()


def _connectors(build, link):
    from aprilcube import mono
    return mono._union(list(build.links[link].mounts.values()))


def test_empty_root_is_dropped(tmp_path):
    urdf = tmp_path / "r.urdf"
    urdf.write_text("""<robot name="r">
      <link name="world"/>
      <link name="a"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
      <link name="b"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
      <joint name="w" type="fixed"><parent link="world"/><child link="a"/><origin xyz="0 0 0.5"/></joint>
      <joint name="j" type="revolute"><parent link="a"/><child link="b"/><origin xyz="0.2 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
    </robot>""")
    robot = twin.drop_empty_links(twin.load_robot(urdf))
    assert robot.root == "a" and set(robot.links) == {"a", "b"}
    assert np.allclose(_joint(robot, "j").origin[:3, 3], [200.0, 0.0, 0.0])


def test_largest_cube_in_a_box():
    import trimesh

    box = trimesh.creation.box(extents=(100.0, 80.0, 60.0))
    edge, pose = twin.largest_cube(twin.Hull.of(box.vertices), samples=20)
    assert edge == pytest.approx(60.0, abs=1e-3)
    assert np.allclose(pose[:3, :3], np.eye(3))


def test_cubes_fit_their_links_and_stay_apart(demo):
    robot, build = demo
    for name, link in build.links.items():
        assert link.cube.overshoot == pytest.approx(0.0, abs=0.05), name
        assert link.cube.edge >= twin.min_module_edge("4x4_250", generate.MonoConfig(enabled=True))
    for joint in robot.joints:
        p, c = build.links[joint.parent].cube, build.links[joint.child].cube
        for value in twin.sweep_values(joint):
            pose = joint.origin @ twin.joint_motion(joint, value) @ c.pose
            assert not twin.boxes_intersect(p.edge, p.pose, c.edge, pose), joint.name
    assert not build.warnings


def test_hinge_moves_freely_and_stops_at_its_limits(demo):
    robot, build = demo
    joint = _joint(robot, "hinge")
    parent, child = _connectors(build, joint.parent), _connectors(build, joint.child)

    def clash(angle):
        return _overlap(parent, twin._place(child, joint.origin @ twin.joint_motion(joint, angle)))

    for angle in np.linspace(joint.lower + 0.02, joint.upper - 0.02, 9):
        assert clash(angle) < 1e-3, angle
    assert clash(joint.upper + 0.25) > 0.05
    assert clash(joint.lower - 0.25) > 0.05


def test_slider_moves_freely_and_stops_at_its_limits(demo):
    robot, build = demo
    joint = _joint(robot, "slide")
    cap = next(part for name, link, part, _ in build.loose if name == "cap slide")
    parent = _connectors(build, joint.parent) + cap
    child = _connectors(build, joint.child)

    def clash(value):
        return _overlap(parent, twin._place(child, joint.origin @ twin.joint_motion(joint, value)))

    for value in np.linspace(joint.lower + 0.1, joint.upper - 0.1, 9):
        assert clash(value) < 1e-3, value
    assert clash(joint.upper + 1.0) > 0.05
    assert clash(joint.lower - 1.0) > 0.05


def test_continuous_joint_has_no_stops(demo):
    robot, build = demo
    joint = _joint(robot, "spin")
    assert build.joints["spin"]["stops"] is None
    parent, child = _connectors(build, joint.parent), _connectors(build, joint.child)
    for angle in np.linspace(-np.pi, np.pi, 13):
        assert _overlap(parent, twin._place(child, joint.origin @ twin.joint_motion(joint, angle))) < 1e-3


def test_bolts_and_nuts_fit_through_the_knuckles(demo):
    robot, build = demo
    for name, link, part, _ in build.loose:
        if not name.startswith(("bolt", "nut")):
            continue
        joint = _joint(robot, name.split()[1])
        parent = _connectors(build, joint.parent)
        child = twin._place(_connectors(build, joint.child), joint.origin)
        assert _overlap(part, parent) < 1e-3 and _overlap(part, child) < 1e-3, name


def test_link_configs_put_corners_on_the_black_border(demo, tmp_path):
    _, build = demo
    twin.write_twin(build, str(tmp_path), "twin_demo.urdf", 1.0)
    seen = set()
    for name, link in build.links.items():
        path = tmp_path / "links" / name / "config.json"
        load_cube_config(str(path))
        config = json.loads(path.read_text())
        lattice = [twin._place(p, link.cube.pose) for p in link.module.black.values()]
        for marker in config["markers"]:
            assert marker["id"] not in seen
            seen.add(marker["id"])
            corners = np.array(marker["corners_mm"])
            center = corners.mean(axis=0)
            normal = np.array(marker["normal"])
            for corner in corners:
                probe = corner + 0.4 * (center - corner) / np.linalg.norm(center - corner) - 0.1 * normal
                from manifold3d import Manifold
                dot = Manifold.cube((0.05, 0.05, 0.05), center=True).translate(tuple(probe))
                assert any(_overlap(dot, part) > 0 for part in lattice), (name, marker["face"])


def test_cli_writes_twin(tmp_path):
    out = tmp_path / "demo"
    twin.main([str(DEMO), "--quick", "-o", str(out)])
    record = json.loads((out / "twin.json").read_text())
    assert record["root"] == "base" and len(record["joints"]) == 4
    assert {j["hardware"] for j in record["joints"]} == {"knuckle", "slider", "strut"}
    for key in ["black", "white", "nets", "connectors", "hardware", "assembly", "preview"]:
        assert (out / twin.TWIN_FILES[key]).exists()
    assert (out / "README.md").read_text().startswith("# Printed twin")
