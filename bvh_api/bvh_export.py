"""HumanIK-style BVH export from per-frame joint positions (MediaPipe world space)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from humanik_mapping import JOINT_ORDER

PARENT: dict[str, str | None] = {
    "Hips": None,
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToe": "LeftFoot",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToe": "RightFoot",
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Neck": "Spine2",
    "Head": "Neck",
    "LeftShoulder": "Spine2",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
}

DFS_ORDER: list[str] = [
    "Hips",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToe",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToe",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
]

CM = 1.0


def _pelvis_rotation(pos: dict[str, np.ndarray]) -> R:
    lh = pos["LeftUpLeg"]
    rh = pos["RightUpLeg"]
    chest = pos["Spine2"]
    hips = pos["Hips"]
    x = lh - rh
    x = x / (np.linalg.norm(x) + 1e-12)
    y = chest - hips
    y = y / (np.linalg.norm(y) + 1e-12)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-12)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    return R.from_matrix(np.stack([x, y, z], axis=1))


def _rot_align(a: np.ndarray, b: np.ndarray) -> R:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return R.identity()
    a = a / na
    b = b / nb
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = np.linalg.norm(v)
    if s < 1e-10:
        return R.identity() if c > 0 else R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    mat = np.eye(3) + k + k @ k * ((1.0 - c) / (s * s))
    return R.from_matrix(mat)


def _rest_rotations(rest: dict[str, np.ndarray]) -> dict[str, R]:
    r_rest: dict[str, R] = {}
    r_rest["Hips"] = _pelvis_rotation(rest)
    for name in DFS_ORDER:
        if name == "Hips":
            continue
        p = PARENT[name]
        assert p is not None
        d = rest[name] - rest[p]
        o = r_rest[p].inv().apply(d)
        nv = np.linalg.norm(o)
        nd = np.linalg.norm(d)
        r_loc = _rot_align(o / (nv + 1e-12), r_rest[p].inv().apply(d) / (nd + 1e-12))
        r_rest[name] = r_rest[p] * r_loc
    return r_rest


def _bind_offsets_m(rest: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "Hips": np.array([-0.001795, -0.223333, 0.028219]),
        "LeftUpLeg": np.array([0.069520, -0.091406, -0.006815]),
        "LeftLeg": np.array([0.034277, -0.375199, -0.004496]),
        "LeftFoot": np.array([-0.013596, -0.397961, -0.043693]),
        "LeftToe": np.array([0.026358, -0.055791, 0.119288]),
        "RightUpLeg": np.array([-0.067670, -0.090522, -0.004320]),
        "RightLeg": np.array([-0.038290, -0.382569, -0.008850]),
        "RightFoot": np.array([0.015774, -0.398415, -0.042312]),
        "RightToe": np.array([-0.025372, -0.048144, 0.123348]),
        "Spine": np.array([-0.002533, 0.108963, -0.026696]),
        "Spine1": np.array([0.005487, 0.135180, 0.001092]),
        "Spine2": np.array([0.001457, 0.052922, 0.025425]),
        "Neck": np.array([-0.002778, 0.213870, -0.042857]),
        "Head": np.array([0.005152, 0.064970, 0.051349]),
        "LeftShoulder": np.array([0.078845, 0.121749, -0.034090]),
        "LeftArm": np.array([0.090977, 0.030469, -0.008868]),
        "LeftForeArm": np.array([0.259612, -0.012772, -0.027456]),
        "LeftHand": np.array([0.249234, 0.008986, -0.001171]),
        "RightShoulder": np.array([-0.081759, 0.118833, -0.038615]),
        "RightArm": np.array([-0.096012, 0.032551, -0.009143]),
        "RightForeArm": np.array([-0.253742, -0.013329, -0.021401]),
        "RightHand": np.array([-0.255298, 0.007772, -0.005559]),
    }


def _r_world_chain(pos: dict[str, np.ndarray], offsets_m: dict[str, np.ndarray]) -> dict[str, R]:
    r_world: dict[str, R] = {}
    r_world["Hips"] = _pelvis_rotation(pos)
    for name in DFS_ORDER:
        if name == "Hips":
            continue
        p = PARENT[name]
        assert p is not None
        rp = r_world[p]
        delta_w = pos[name] - pos[p]
        v_parent = rp.inv().apply(delta_w)
        o = offsets_m[name]
        no = np.linalg.norm(o)
        nv = np.linalg.norm(v_parent)
        if no < 1e-10 or nv < 1e-10:
            r_local = R.identity()
        else:
            r_local = _rot_align(o / no, v_parent / nv)
        r_world[name] = rp * r_local
    return r_world


def _local_from_world(r_world: dict[str, R]) -> dict[str, R]:
    """BVH motion stores local rotation per joint; root is world space."""
    r_local: dict[str, R] = {}
    r_local["Hips"] = r_world["Hips"]
    for name in DFS_ORDER:
        if name == "Hips":
            continue
        p = PARENT[name]
        assert p is not None
        r_local[name] = r_world[p].inv() * r_world[name]
    return r_local


def _emit_hierarchy(rest: dict[str, np.ndarray], offsets_m: dict[str, np.ndarray]) -> str:
    lines: list[str] = []

    def emit(name: str, indent: int) -> None:
        pad = "\t" * indent
        par = PARENT[name]
        children = [j for j in DFS_ORDER if PARENT.get(j) == name]
        if par is None:
            o = offsets_m[name] * CM
            lines.append(f"{pad}ROOT {name}")
            lines.append(f"{pad}{{")
            lines.append(f"{pad}\tOFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
            lines.append(f"{pad}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
        else:
            o = offsets_m[name] * CM
            lines.append(f"{pad}JOINT {name}")
            lines.append(f"{pad}{{")
            lines.append(f"{pad}\tOFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
            lines.append(f"{pad}\tCHANNELS 3 Zrotation Yrotation Xrotation")

        for ch in children:
            emit(ch, indent + 1)

        if not children:
            if par is not None:
                lines.append(f"{pad}\tEnd Site")
                lines.append(f"{pad}\t{{")
                lines.append(f"{pad}\t\tOFFSET 0.000000 0.000000 0.000000")
                lines.append(f"{pad}\t}}")
        lines.append(f"{pad}}}")

    emit("Hips", 0)
    return "HIERARCHY\n" + "\n".join(lines)


def positions_to_bvh(
    frames: list[dict[str, np.ndarray]],
    fps: float,
    rest: dict[str, np.ndarray] | None = None,
) -> str:
    if not frames:
        raise ValueError("no frames")
    rest = rest or frames[0]
    offsets_m = _bind_offsets_m(rest)
    hier = _emit_hierarchy(rest, offsets_m)
    nframes = len(frames)
    dt = 1.0 / max(fps, 1e-6)
    motion_lines = [f"Frames: {nframes}", f"Frame Time: {dt:.6f}"]

    for fr in frames:
        r_w = _r_world_chain(fr, offsets_m)
        r_l = _local_from_world(r_w)
        parts: list[float] = []
        h = fr["Hips"] * CM
        rh = r_l["Hips"]
        ez = rh.as_euler("zyx", degrees=True)
        parts.extend([h[0], h[1], h[2], ez[0], ez[1], ez[2]])
        for name in DFS_ORDER:
            if name == "Hips":
                continue
            e = r_l[name].as_euler("zyx", degrees=True)
            parts.extend([e[0], e[1], e[2]])
        motion_lines.append(" ".join(f"{v:.6f}" for v in parts))

    return hier + "\nMOTION\n" + "\n".join(motion_lines) + "\n"
