"""Cluttered tabletop scene with multiple pegs and multiple sockets.

Extends the single-peg ``scene_cfg`` to the matching-required regime:

  - N_PEGS pegs of identical size but distinct color, scattered across
    one side of the worktable
  - N_HOLES sockets on the other side, each tinted the color of its
    intended peg and given a different clearance (insertion tolerance)
  - A fixed third-person workbench camera looking down at the table,
    used for the one-shot perception call at episode start
  - The correct peg→socket assignment is by COLOR; the swept clearance
    makes each colored pair a different-difficulty insertion, which is
    the accuracy-vs-tolerance signal the benchmark reports

All shared constants (peg geometry, friction, gripper helpers, the
``insertion_success`` check, the hand-frame offset math) are imported
from ``scene_cfg`` so a single source of truth governs both scenes.

Motion-procedure assumption (TCP control)
-----------------------------------------
The runner that consumes this scene drives the Panda HAND FRAME (TCP)
directly via a DMP, not the peg tip. So the helpers exposed here serve
a slightly different role than in the single-peg version:

  - ``peg_place_pos_w(peg_id)``  — where a given peg STARTS on the
    table (and stays, since the runner doesn't warp pegs to a canonical
    pick spot anymore — it navigates to wherever each peg actually
    sits). The DMP segment-2 GOAL is computed via
    ``hand_pos_for_grasp(perceived_peg_center)``.

  - ``hole_top_w(hole_id)`` / ``hole_center_w(hole_id)``  — the rim and
    seat positions, used together with ``hand_pos_for_peg_tip`` to
    compute the TCP-space INSERTION GOAL.

The scene config itself doesn't know about TCP or DMP — it just lays out
N pegs and N sockets and gives them stable IDs. The TCP-vs-peg story
lives entirely in the runner.
"""
from __future__ import annotations

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# Reuse all shared constants and helpers — single source of truth.
from .scene_cfg import (  # noqa: F401
    TABLE_TOP_Z,
    TABLE_SIZE_X, TABLE_SIZE_Y, TABLE_THICKNESS,
    PEG_LENGTH, PEG_RADIUS, HOLE_DEPTH,
    PEG_FRICTION_STATIC, PEG_FRICTION_DYNAMIC, PEG_RESTITUTION,
    SOCKET_BLOCK_HEIGHT, SOCKET_BLOCK_SIZE_XY,
    GRIPPER_DOWN_QUAT,
    PEG_BODY_CENTER_HAND_Z, PEG_TIP_OFFSET_Z,
    finger_grip_target,
    hand_pos_for_grasp, hand_pos_for_peg_tip,
    peg_tip_from_body, peg_pose_from_hand,
    insertion_success,
)


# ─── Cluttered scene parameters ──────────────────────────────────────────────
# Design: this experiment measures INSERTION ACCURACY (the residual lateral /
# depth error each method achieves), NOT size discrimination. So every peg
# shares ONE diameter; the only per-peg variation is COLOR, which carries
# semantic meaning for the high-level planner — it decides which peg goes to
# which socket by matching colors. Pose estimation therefore only needs the
# peg/hole POSITION (robust), never an accurate diameter (hard at this scale).
PEG_DIAMETER = 2 * PEG_RADIUS        # uniform; single source of truth via PEG_RADIUS

PEG_SPECS: list[dict] = [
    # (id, diameter, color, place_xy) — diameter uniform, color is the identity
    {"id": "peg_0", "diameter": PEG_DIAMETER, "color": (0.85, 0.30, 0.20), "xy": (0.40, -0.25)},
    {"id": "peg_1", "diameter": PEG_DIAMETER, "color": (0.20, 0.70, 0.30), "xy": (0.50, -0.18)},
    {"id": "peg_2", "diameter": PEG_DIAMETER, "color": (0.20, 0.40, 0.90), "xy": (0.45, -0.10)},
]

# Sockets carry the COLOR of their intended peg (the whole block is tinted)
# so the planner matches by color, plus a swept CLEARANCE so a single uniform
# peg sees a range of insertion tolerances — the accuracy-vs-tolerance axis of
# the benchmark. ``clearance`` is the DIAMETER gap (hole_Ø − peg_Ø); the rim
# radius is derived from it. Each socket's color mirrors the same-index peg.
SOCKET_SPECS: list[dict] = [
    # (id, color, clearance[m], xy)
    {"id": "hole_0", "color": (0.85, 0.30, 0.20), "clearance": 0.010, "xy": (0.45, 0.10)},  # red  ↔ peg_0, 10 mm
    {"id": "hole_1", "color": (0.20, 0.70, 0.30), "clearance": 0.005, "xy": (0.50, 0.20)},  # green↔ peg_1,  5 mm
    {"id": "hole_2", "color": (0.20, 0.40, 0.90), "clearance": 0.001, "xy": (0.40, 0.28)},  # blue ↔ peg_2,  1 mm
]
# Derive each rim radius from the shared peg diameter + the socket's clearance.
for _s in SOCKET_SPECS:
    _s["hole_radius"] = (PEG_DIAMETER + _s["clearance"]) / 2.0

# Standard hole geometry: hole-radius is per-socket, but depth and rim
# are common. We KEEP the rim-of-cuboids construction from the single-
# socket scene because it gives clean contact geometry without CSG.
HOLE_DEPTH_M = HOLE_DEPTH


# ─── Camera (workbench third-person view) ────────────────────────────────────
# Mount the camera high and tilted slightly toward the work area, so it
# sees both peg-side and socket-side of the table. The pose is in world
# frame; the runner reads the camera's T_world_cam after sim startup
# (Isaac Lab exposes ``camera.data.pos_w`` / ``quat_w_world``).
CAMERA_POS_W = (0.85, 0.0, 0.85)
# Pointing nominally toward (0.45, 0.0, 0.0). The orientation is built
# directly in cluttered_run_e2e using a look-at helper, so we don't fix
# a quaternion here.
CAMERA_TARGET_W = (0.45, 0.0, 0.05)

CAMERA_W = 640
CAMERA_H = 480
CAMERA_FOCAL_LEN = 24.0       # mm; Isaac Lab CameraCfg uses USD units
CAMERA_APERTURE = 20.955      # default Kit aperture (matches a 1" sensor)


# ─── Helpers for the runner ──────────────────────────────────────────────────
def all_peg_ids() -> list[str]:
    return [s["id"] for s in PEG_SPECS]


def all_hole_ids() -> list[str]:
    return [s["id"] for s in SOCKET_SPECS]


def peg_spec(peg_id: str) -> dict:
    for s in PEG_SPECS:
        if s["id"] == peg_id:
            return s
    raise KeyError(f"no PEG_SPEC with id={peg_id}")


def socket_spec(hole_id: str) -> dict:
    for s in SOCKET_SPECS:
        if s["id"] == hole_id:
            return s
    raise KeyError(f"no SOCKET_SPEC with id={hole_id}")


def peg_place_pos_w(peg_id: str) -> np.ndarray:
    """World position the peg occupies on the table (standing upright).

    This is the peg's BODY CENTER in world frame: ``(xy_spec, z_top)``
    where ``z_top = TABLE_TOP_Z + PEG_LENGTH/2``. In the TCP-controlled
    procedure the runner does NOT warp pegs to a canonical pick spot —
    it navigates the TCP to wherever each peg sits per the PEG_SPECS
    table. Used by:

      - the runner's ``reset_to_home`` to write each peg's initial pose
        (zero velocity, identity orientation) at episode start
      - the ``GroundTruthPerception`` backend, which uses these
        positions in place of a measured pose when no VLM is in the loop
    """
    s = peg_spec(peg_id)
    return np.array([s["xy"][0], s["xy"][1], TABLE_TOP_Z + PEG_LENGTH / 2],
                    dtype=np.float32)


def peg_color(peg_id: str) -> tuple[float, float, float]:
    """RGB color of a peg — the planner's identity / matching key."""
    return tuple(peg_spec(peg_id)["color"])


def socket_color(hole_id: str) -> tuple[float, float, float]:
    """RGB color a socket is tinted with (= its intended peg's color)."""
    return tuple(socket_spec(hole_id)["color"])


def socket_clearance(hole_id: str) -> float:
    """Diameter clearance (hole_Ø − peg_Ø) for this socket, in meters."""
    return float(socket_spec(hole_id)["clearance"])


def socket_top_z(hole_id: str) -> float:
    """World Z of the rim top for this socket (peg-tip target for seating)."""
    return TABLE_TOP_Z + SOCKET_BLOCK_HEIGHT + HOLE_DEPTH_M


def hole_center_w(hole_id: str) -> np.ndarray:
    """World XYZ of the hole bottom (where a fully seated peg tip would land)."""
    s = socket_spec(hole_id)
    z = TABLE_TOP_Z + SOCKET_BLOCK_HEIGHT
    return np.array([s["xy"][0], s["xy"][1], z], dtype=np.float32)


def hole_top_w(hole_id: str) -> np.ndarray:
    """World XYZ of the hole rim top (DMP insertion goal for the tip)."""
    s = socket_spec(hole_id)
    return np.array([s["xy"][0], s["xy"][1], socket_top_z(hole_id)],
                    dtype=np.float32)


# ─── Camera look-at quaternion ───────────────────────────────────────────────
# Defined BEFORE the ClutteredInsertionSceneCfg class body, because the
# class body calls _camera_lookat_quat() at class-definition time (to bake
# the camera's rest orientation into the CameraCfg). Python evaluates
# class bodies top-to-bottom at import time, so the helper has to exist
# at the moment the class statement runs.
def _mat_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (float(w), float(x), float(y), float(z))


def _camera_lookat_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (w, x, y, z) for a camera at ``eye`` looking at ``target``.

    OpenGL convention: camera looks down -Z, +Y is up. We return the
    quaternion that rotates the world basis into the camera basis.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    f = target - eye
    f /= (np.linalg.norm(f) + 1e-9)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        # Singular when looking straight up/down; pick an arbitrary right axis.
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r) + 1e-9
    u = np.cross(r, f)

    # Camera basis: +X = right (r), +Y = up (u), -Z = forward (f).
    R = np.stack([r, u, -f], axis=1)        # world -> cam-OpenGL basis cols
    return _mat_to_quat(R)


# ─── Analytic camera intrinsics / extrinsics ─────────────────────────────────
# The workbench camera is STATIC (fixed offset, never moves), so its K and
# world pose are fully determined by the constants above. We compute them
# analytically rather than reading ``camera.data`` because that buffer is
# unpopulated at perception time for this fixed-offset camera (pos_w = origin,
# quat = NaN/zero), which otherwise corrupts the back-projection extrinsic.
def workbench_camera_K() -> np.ndarray:
    """3x3 pinhole intrinsics from focal length / aperture / resolution."""
    fx = CAMERA_FOCAL_LEN * CAMERA_W / CAMERA_APERTURE
    fy = fx                       # square pixels (vertical aperture scales with H/W)
    cx = CAMERA_W / 2.0
    cy = CAMERA_H / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def workbench_camera_T_world_cam() -> np.ndarray:
    """4x4 camera→world extrinsic in the ROS/pinhole convention.

    Columns of the rotation are the world directions of the camera axes:
    +X right, +Y down, +Z forward (into the scene) — exactly the frame
    ``perception.pose_estimation.back_project`` assumes. Built from the static
    look-at (CAMERA_POS_W → CAMERA_TARGET_W).
    """
    C = np.asarray(CAMERA_POS_W, dtype=np.float64)
    tgt = np.asarray(CAMERA_TARGET_W, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0])
    fwd = tgt - C
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-9
    cam_up = np.cross(right, fwd)         # world +up of the camera
    R = np.stack([right, -cam_up, fwd], axis=1)   # X=right, Y=down, Z=forward
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = C
    return T


# ─── The cluttered scene config ──────────────────────────────────────────────
@configclass
class ClutteredInsertionSceneCfg(InteractiveSceneCfg):
    """Franka + N pegs + N socket fixtures + workbench camera."""

    # ----- Ground + lights -----
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # ----- Franka stand + worktable (same as the single-peg scene) -----
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
        ),
    )
    worktable = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WorkTable",
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_SIZE_X, TABLE_SIZE_Y, TABLE_THICKNESS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.4, 0.35)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            activate_contact_sensors=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.45, 0.0, TABLE_TOP_Z - TABLE_THICKNESS / 2),
        ),
    )

    # ----- Franka -----
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0
    robot.spawn.rigid_props.disable_gravity = True
    robot.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=PEG_FRICTION_STATIC,
        dynamic_friction=PEG_FRICTION_DYNAMIC,
        restitution=PEG_RESTITUTION,
    )
    robot.init_state.joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.569,
        "panda_joint3": 0.0,
        "panda_joint4": -2.810,
        "panda_joint5": 0.0,
        "panda_joint6": 3.037,
        "panda_joint7": 0.741,
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    }

    # ----- Workbench camera (RGB + depth) -----
    workbench_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/WorkbenchCamera",
        update_period=0.0,                  # only sampled on-demand
        height=CAMERA_H,
        width=CAMERA_W,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LEN,
            focus_distance=400.0,
            horizontal_aperture=CAMERA_APERTURE,
            clipping_range=(0.05, 4.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=CAMERA_POS_W,
            # Quaternion oriented to look from CAMERA_POS_W toward
            # CAMERA_TARGET_W (computed offline once; see camera_lookat
            # helper below if you want to regenerate it).
            rot=_camera_lookat_quat(CAMERA_POS_W, CAMERA_TARGET_W),
            convention="opengl",
        ),
    )

    # ----- Pegs (built procedurally below in __post_init__) -----
    # Each peg gets its own RigidObjectCfg attribute named ``peg_<i>``.
    # We can't do this in a class body directly because configclass doesn't
    # accept dynamic attributes, so we attach them via the runner's
    # initialization sequence (see _attach_pegs / _attach_sockets below).


# ─── Procedural attachment of pegs + sockets to the cfg ──────────────────────
def attach_pegs_and_sockets(cfg: ClutteredInsertionSceneCfg) -> ClutteredInsertionSceneCfg:
    """Dynamically attach the N pegs and N sockets to a config instance.

    ``configclass`` doesn't take per-instance attributes well, but the
    InteractiveScene walks ``__dict__`` for AssetBaseCfg / RigidObjectCfg
    fields, so setting them as attributes after construction works in
    practice. Call this once before passing the cfg to InteractiveScene.
    """
    # Pegs (dynamic rigid bodies, standing on the table at their place_xy)
    for spec in PEG_SPECS:
        pid = spec["id"]
        diameter = spec["diameter"]
        radius = diameter / 2.0
        x, y = spec["xy"]
        z = TABLE_TOP_Z + PEG_LENGTH / 2
        peg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{_camel(pid)}",
            spawn=sim_utils.CylinderCfg(
                radius=radius,
                height=PEG_LENGTH,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=spec["color"],
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=PEG_FRICTION_STATIC,
                    dynamic_friction=PEG_FRICTION_DYNAMIC,
                    restitution=PEG_RESTITUTION,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=1.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, z)),
        )
        setattr(cfg, pid, peg)

    # Sockets: a base block + 4 rim walls per socket. Built as KINEMATIC
    # RigidObjectCfgs (not AssetBaseCfg) so a runner can teleport a socket to a
    # new XY each reset via write_root_pose_to_sim — see
    # socket_component_offsets(). Kinematic bodies still collide with the
    # dynamic peg but are unaffected by contact, so static-scene consumers
    # (e.g. cluttered_run_e2e, which never moves them) behave exactly as before.
    for spec in SOCKET_SPECS:
        _attach_socket(cfg, spec["id"], spec["color"])

    return cfg


def _socket_layout(hole_id: str):
    """Single source of truth for the 5 prims that make up one socket.

    Returns a list of ``(key, size_xyz, dx, dy, z_center, is_base)`` where
    ``key`` is the per-component suffix (``base``/``rim_north``/…), ``dx``/``dy``
    are the component's XY offset from the socket CENTER, and ``z_center`` is
    its world Z. Both ``_attach_socket`` (spawn) and ``socket_component_offsets``
    (runtime teleport) derive from this, so the spawn geometry and the move
    geometry can never drift apart.
    """
    s = socket_spec(hole_id)
    hole_radius = s["hole_radius"]
    base_z = TABLE_TOP_Z
    block_top_z = base_z + SOCKET_BLOCK_HEIGHT
    wall_z = block_top_z + HOLE_DEPTH_M / 2
    base_center_z = base_z + SOCKET_BLOCK_HEIGHT / 2
    outer_half = SOCKET_BLOCK_SIZE_XY / 2
    inner_half = hole_radius
    wall_thickness = max(0.005, (SOCKET_BLOCK_SIZE_XY - 2 * hole_radius) / 2)
    wall_offset = (outer_half + inner_half) / 2
    long_size = (SOCKET_BLOCK_SIZE_XY, wall_thickness, HOLE_DEPTH_M)
    short_size = (wall_thickness, 2 * inner_half, HOLE_DEPTH_M)
    base_size = (SOCKET_BLOCK_SIZE_XY, SOCKET_BLOCK_SIZE_XY, SOCKET_BLOCK_HEIGHT)
    return [
        ("base", base_size, 0.0, 0.0, base_center_z, True),
        ("rim_north", long_size, 0.0, +wall_offset, wall_z, False),
        ("rim_south", long_size, 0.0, -wall_offset, wall_z, False),
        ("rim_east", short_size, +wall_offset, 0.0, wall_z, False),
        ("rim_west", short_size, -wall_offset, 0.0, wall_z, False),
    ]


def socket_component_offsets(hole_id: str):
    """``[(scene_key, dx, dy, z_center), ...]`` for a socket's 5 rigid prims.

    To move a socket to world XY ``(cx, cy)``, write each component's root pose
    to ``(cx + dx, cy + dy, z_center)`` (identity orientation). ``scene_key`` is
    the InteractiveScene key (= the cfg attribute name) for that component, so
    ``scene[scene_key].write_root_pose_to_sim(...)`` addresses it directly.
    """
    return [(f"{hole_id}_{key}", dx, dy, zc)
            for key, _size, dx, dy, zc, _is_base in _socket_layout(hole_id)]


def _attach_socket(cfg, hid, color):
    """Attach one socket fixture (base + 4 rim walls) named after hid.

    The whole block — base + rim walls — is tinted ``color`` (the intended
    peg's color) so the high-level planner can match peg→socket by color.
    The rims are tinted a touch darker than the base purely as a depth cue so
    the hole opening stays readable; both are the socket's identity color.

    Each prim is a KINEMATIC RigidObjectCfg (movable at runtime, immovable by
    contact). Component geometry/offsets come from ``_socket_layout``.
    """
    rim_color = tuple(0.8 * c for c in color)
    name = _camel(hid)
    x, y = socket_spec(hid)["xy"]
    for key, size, dx, dy, zc, is_base in _socket_layout(hid):
        prim_suffix = "Base" if is_base else "Rim" + key.split("_")[1].capitalize()
        comp = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}{prim_suffix}",
            spawn=sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(color) if is_base else rim_color),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                activate_contact_sensors=False,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x + dx, y + dy, zc)),
        )
        setattr(cfg, f"{hid}_{key}", comp)


def _camel(s: str) -> str:
    return "".join(p.capitalize() for p in s.split("_"))
