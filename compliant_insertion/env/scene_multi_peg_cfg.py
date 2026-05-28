"""Scene configuration for a FOUR-peg / FOUR-hole insertion task.

This is the multi-socket generalisation of the original single peg-in-hole
scene. The world still has a Franka Panda on a stand and a flat worktable in
front of it, but now:

  * Four identical pegs stand upright on the table at four reachable pick
    locations (they look the same — same radius, length, colour).
  * One socket block in the MIDDLE of the table holds four holes, each with a
    DIFFERENT clearance / tolerance:
        hole 0 : 10  mm clearance  (very loose)
        hole 1 :  5  mm clearance  (loose)
        hole 2 :  1  mm clearance  (tight)
        hole 3 :  0.1 mm clearance (very tight)
    The four holes are laid out in a 2x2 grid inside the socket block.

The runner picks each peg in turn, transports it to the matching hole, and
attempts the insertion, then moves on to the next.

Peg attachment / grasp model
----------------------------
Each peg is a normal DYNAMIC RigidObjectCfg standing on the table. The runner
drives the empty gripper down onto the peg, closes the fingers, and from then
on the peg is held purely by fingertip friction (no kinematic tracking). This
matches the original single-peg scene and keeps a real grasp/slip signal.

TCP control (NEW)
-----------------
The original scene commanded the peg TIP and converted to a hand target via
``hand_pos_for_peg_tip``. This version controls the TCP (tool centre point,
the tip of the closed fingers, ``PANDA_FINGERTIP_OFFSET_Z`` along hand +Z)
directly. The DMP therefore lives in TCP-world coordinates and the only
hand<->TCP conversion is a single constant offset along the down-pointing
gripper axis. Helpers ``hand_pos_for_tcp`` / ``tcp_from_hand`` implement that.

Frame conventions and most geometry are unchanged from the single-peg scene;
see the original file's docstring for the full derivation of the peg/hand
offsets.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # noqa: I001


# ─── Peg geometry (all four pegs identical). ─────────────────────────────────
PEG_RADIUS = 0.008
PEG_LENGTH = 0.05

NUM_TASKS = 4   # four pegs, four holes

# ─── Hole tolerances (one per hole). ─────────────────────────────────────────
# Clearance per hole, in metres. hole_radius = peg_radius + clearance.
# Ordered loosest -> tightest so task 0 is the easy one.
HOLE_CLEARANCES = (0.010, 0.005, 0.001, 0.0001)   # 10mm, 5mm, 1mm, 0.1mm
HOLE_RADII = tuple(PEG_RADIUS + c for c in HOLE_CLEARANCES)
HOLE_DEPTH = 0.04             # how deep a peg can sink before bottoming out

# ─── Socket block (single block in the middle holding all four holes). ───────
# The block is a flat slab; the four holes sit in a 2x2 grid on top of it,
# each surrounded by its own four rim walls.
SOCKET_BLOCK_HEIGHT = 0.01    # 1 cm thick base under the rims
RIM_HEIGHT = HOLE_DEPTH

# Spacing between adjacent hole centres in the 2x2 grid.
HOLE_PITCH = 0.06             # 6 cm between hole centres
# Each hole gets a square cell of this half-size for its rim walls.
CELL_HALF = HOLE_PITCH / 2    # 3 cm — walls of neighbouring cells abut

# Overall socket block footprint must cover the 2x2 grid of cells.
SOCKET_BLOCK_SIZE_XY = 2 * HOLE_PITCH   # 12 cm square, comfortably covers grid

# ─── Placement. ──────────────────────────────────────────────────────────────
TABLE_TOP_Z = 0.0

# ----- Socket location: MIDDLE of the table, in front of the robot. -----
SOCKET_X = 0.45
SOCKET_Y = 0.0
SOCKET_BASE_Z = TABLE_TOP_Z

# Layered Z stack of the socket fixture.
SOCKET_BLOCK_TOP_Z = SOCKET_BASE_Z + SOCKET_BLOCK_HEIGHT
SOCKET_TOP_Z = SOCKET_BLOCK_TOP_Z + RIM_HEIGHT   # top of the rim walls

# ----- Hole centres (2x2 grid around the socket centre). -----
# (x, y) of each hole's axis, on the table. Index order matches HOLE_CLEARANCES.
_HALF_PITCH = HOLE_PITCH / 2
HOLE_CENTERS_XY = (
    (SOCKET_X - _HALF_PITCH, SOCKET_Y + _HALF_PITCH),   # hole 0 (10 mm)
    (SOCKET_X + _HALF_PITCH, SOCKET_Y + _HALF_PITCH),   # hole 1 (5 mm)
    (SOCKET_X - _HALF_PITCH, SOCKET_Y - _HALF_PITCH),   # hole 2 (1 mm)
    (SOCKET_X + _HALF_PITCH, SOCKET_Y - _HALF_PITCH),   # hole 3 (0.1 mm)
)

# Peg-tip goal (rim top) and seated point (block top) per hole.
HOLE_CENTERS_TOP = tuple(
    (x, y, SOCKET_TOP_Z) for (x, y) in HOLE_CENTERS_XY
)
HOLE_CENTERS_BOTTOM = tuple(
    (x, y, SOCKET_BLOCK_TOP_Z) for (x, y) in HOLE_CENTERS_XY
)

# ----- Peg pick locations (four reachable spots on the table). -----
# Spread along the near edge (small +X, spanning -Y..+Y) so they are clearly
# separated from the central socket and easily reachable. Move freely.
PEG_PLACE_Z = TABLE_TOP_Z + PEG_LENGTH / 2   # standing upright -> centre is +half
PEG_PLACE_XY = (
    (0.40, -0.30),   # peg 0
    (0.55, -0.30),   # peg 1
    (0.40,  0.30),   # peg 2
    (0.55,  0.30),   # peg 3
)
PEG_PLACE_POSITIONS = tuple(
    (x, y, PEG_PLACE_Z) for (x, y) in PEG_PLACE_XY
)
#print(f"[multi peg config] PEG_PLACE_POSITIONS: {PEG_PLACE_POSITIONS}")
# Table slab dimensions (a thin visual+collision box under the work surface).
TABLE_SIZE_X = 0.4
TABLE_SIZE_Y = 1.0
TABLE_THICKNESS = 0.02

# ─── Peg-in-gripper attachment geometry (unchanged from single-peg scene). ───
PANDA_FINGERTIP_OFFSET_Z = 0.1034   # panda_hand -> TCP (tip of closed fingers)
PANDA_PAD_CENTER_OFFSET_Z = 0.1     # panda_hand -> centre of finger-pad contact

GRIP_FRACTION = 0.15
_grip_from_top = GRIP_FRACTION * PEG_LENGTH
PEG_BODY_CENTER_HAND_Z = PANDA_PAD_CENTER_OFFSET_Z + (PEG_LENGTH / 2 - _grip_from_top)

PEG_TIP_OFFSET_Z = PEG_BODY_CENTER_HAND_Z + PEG_LENGTH / 2
PEG_BASE_OFFSET_Z = PEG_BODY_CENTER_HAND_Z - PEG_LENGTH / 2

PEG_TIP_OFFSET_HAND = (0.0, 0.0, PEG_TIP_OFFSET_Z)
PEG_BASE_OFFSET_HAND = (0.0, 0.0, PEG_BASE_OFFSET_Z)

# TCP offset along hand +Z (the working point we now command directly).
TCP_OFFSET_HAND = (0.0, 0.0, PANDA_FINGERTIP_OFFSET_Z)

# How far the peg tip sticks out past the TCP (constant for the held peg).
# tip is PEG_TIP_OFFSET_Z along +Z, TCP is PANDA_FINGERTIP_OFFSET_Z along +Z,
# so the tip is this much further along +Z than the TCP.
TIP_BEYOND_TCP_Z = PEG_TIP_OFFSET_Z - PANDA_FINGERTIP_OFFSET_Z

# ─── Grasp / friction parameters (unchanged). ────────────────────────────────
PEG_FRICTION_STATIC = 1.2
PEG_FRICTION_DYNAMIC = 1.0
PEG_RESTITUTION = 0.0

FINGER_SQUEEZE = 0.0015
FINGER_GRIP_POS = max(PEG_RADIUS - FINGER_SQUEEZE, 0.0)


def finger_grip_target(num_envs, num_finger_joints, device, dtype=None):
    """Joint-position target [N, num_finger_joints] that grips the peg."""
    import torch

    return torch.full(
        (num_envs, num_finger_joints),
        FINGER_GRIP_POS,
        device=device,
        dtype=dtype if dtype is not None else torch.float32,
    )


def peg_tip_from_body(peg_pos_w, peg_quat_w):
    """Tip position from a DYNAMIC peg's own body pose (tip = +half along +Z)."""
    import torch
    from isaaclab.utils.math import quat_apply

    half_len = torch.tensor(
        [0.0, 0.0, PEG_LENGTH / 2],
        device=peg_pos_w.device,
        dtype=peg_pos_w.dtype,
    ).expand_as(peg_pos_w)
    return peg_pos_w + quat_apply(peg_quat_w, half_len)


# ─── Down-pointing gripper orientation (base frame). ─────────────────────────
GRIPPER_DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)   # (w, x, y, z): hand +Z -> world -Z


def hand_pos_for_grasp(peg_center_pos_w, hand_quat_w):
    """Hand position (world) to grasp a peg whose CENTRE is at peg_center_pos_w."""
    import torch
    from isaaclab.utils.math import quat_apply

    offset_hand = torch.tensor(
        [0.0, 0.0, PEG_BODY_CENTER_HAND_Z],
        device=peg_center_pos_w.device,
        dtype=peg_center_pos_w.dtype,
    ).expand_as(peg_center_pos_w)
    offset_w = quat_apply(hand_quat_w, offset_hand)
    return peg_center_pos_w - offset_w


# ─── TCP <-> hand conversion (NEW: we control the TCP directly). ─────────────
def hand_pos_for_tcp(tcp_pos_w, hand_quat_w):
    """Hand origin (world) that places the TCP at tcp_pos_w.

        hand_pos_w = tcp_pos_w - R(hand_quat_w) @ (0, 0, PANDA_FINGERTIP_OFFSET_Z)
    """
    import torch
    from isaaclab.utils.math import quat_apply

    off = torch.tensor(
        list(TCP_OFFSET_HAND),
        device=tcp_pos_w.device,
        dtype=tcp_pos_w.dtype,
    ).expand_as(tcp_pos_w)
    return tcp_pos_w - quat_apply(hand_quat_w, off)


def tcp_from_hand(hand_pos_w, hand_quat_w):
    """TCP position (world) from the hand pose (forward of hand_pos_for_tcp)."""
    import torch
    from isaaclab.utils.math import quat_apply

    off = torch.tensor(
        list(TCP_OFFSET_HAND),
        device=hand_pos_w.device,
        dtype=hand_pos_w.dtype,
    ).expand_as(hand_pos_w)
    return hand_pos_w + quat_apply(hand_quat_w, off)


def tcp_goal_for_hole_seat(hole_idx, seat_tip_z=None):
    """TCP-world goal so the peg TIP seats at ``seat_tip_z`` over hole ``hole_idx``.

    For the down-pointing gripper the tip is ``TIP_BEYOND_TCP_Z`` BELOW the TCP
    (peg points down). So to put the tip at height ``seat_tip_z`` the TCP must
    sit ``TIP_BEYOND_TCP_Z`` ABOVE it:

        tcp_z = seat_tip_z + TIP_BEYOND_TCP_Z

    Default seat target is the rim top (touch, not crash through). This is the
    fix for the original "goal too low -> peg crashes into the socket": the goal
    is computed in TCP space from the tip seat height, not by feeding a rim-top
    Z straight into a tip-space DMP.
    """
    x, y = HOLE_CENTERS_XY[hole_idx]
    if seat_tip_z is None:
        seat_tip_z = SOCKET_TOP_Z      # tip rests at rim top
    return (x, y, seat_tip_z + TIP_BEYOND_TCP_Z)


# ─── Geometric success criterion (per-hole radius). ──────────────────────────
def insertion_success(
    tip_pos_w,
    hole_idx,
    depth_threshold=0.5,
):
    """Geometric peg-in-hole success check against hole ``hole_idx``.

    Returns ``(success, lateral_err, depth_frac)`` for the given hole's
    centre and clearance radius. Same semantics as the single-hole version.
    """
    import torch

    cx, cy = HOLE_CENTERS_XY[hole_idx]
    hole_center = torch.tensor(
        [cx, cy, SOCKET_BLOCK_TOP_Z],
        device=tip_pos_w.device, dtype=tip_pos_w.dtype,
    )
    hole_radius = HOLE_RADII[hole_idx]
    rim_top_z = SOCKET_TOP_Z

    lateral_err = torch.norm(tip_pos_w[..., :2] - hole_center[:2], dim=-1)
    rim_height = rim_top_z - SOCKET_BLOCK_TOP_Z
    depth_below_rim = rim_top_z - tip_pos_w[..., 2]
    depth_frac = depth_below_rim / rim_height

    laterally_inside = lateral_err <= hole_radius
    deep_enough = depth_frac >= depth_threshold
    success = laterally_inside & deep_enough
    return success, lateral_err, depth_frac


# ─── Rim-wall geometry helpers (one set of four walls per hole). ─────────────
_WALL_Z = SOCKET_BLOCK_TOP_Z + RIM_HEIGHT / 2   # vertical centre of every rim


def _rim_walls_for_hole(prefix, cx, cy, hole_radius):
    """Build the four AssetBaseCfg rim walls bounding one hole.

    Returns a dict ``{attr_name: AssetBaseCfg}``. Each hole gets a square cell
    of half-size ``CELL_HALF``; the void left in the middle is the bounding
    square of the cylindrical hole (side = 2*hole_radius).
    """
    outer_half = CELL_HALF
    inner_half = hole_radius
    wall_thickness = outer_half - inner_half      # auto-fit to this hole
    wall_offset = (outer_half + inner_half) / 2   # centre of each slab

    size_long = (2 * outer_half, wall_thickness, RIM_HEIGHT)   # north/south
    size_short = (wall_thickness, 2 * inner_half, RIM_HEIGHT)  # east/west
    color = (0.2, 0.2, 0.6)

    def _wall(name, size, pos):
        return AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            spawn=sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                activate_contact_sensors=False,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        )

    return {
        f"{prefix}_north": _wall(
            f"{prefix}_RimNorth", size_long, (cx, cy + wall_offset, _WALL_Z)),
        f"{prefix}_south": _wall(
            f"{prefix}_RimSouth", size_long, (cx, cy - wall_offset, _WALL_Z)),
        f"{prefix}_east": _wall(
            f"{prefix}_RimEast", size_short, (cx + wall_offset, cy, _WALL_Z)),
        f"{prefix}_west": _wall(
            f"{prefix}_RimWest", size_short, (cx - wall_offset, cy, _WALL_Z)),
    }


def _peg_cfg(idx, pos):
    """One dynamic upright peg (all four identical except spawn position)."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + f"Peg{idx}",
        spawn=sim_utils.CylinderCfg(
            radius=PEG_RADIUS,
            height=PEG_LENGTH,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.6, 0.2)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


@configclass
class InsertionSceneCfg(InteractiveSceneCfg):
    """Franka + one socket block with four holes + four pegs on a tabletop."""

    # ----- Ground -----
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # ----- Lighting -----
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # ----- Tabletop mount for the Franka -----
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
        ),
    )

    # ----- Work surface -----
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

    # ----- Single socket base block in the middle (holds all four holes) -----
    socket_base = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SocketBase",
        spawn=sim_utils.CuboidCfg(
            size=(SOCKET_BLOCK_SIZE_XY, SOCKET_BLOCK_SIZE_XY, SOCKET_BLOCK_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            activate_contact_sensors=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(SOCKET_X, SOCKET_Y, SOCKET_BASE_Z + SOCKET_BLOCK_HEIGHT / 2),
        ),
    )

    # ----- Rim walls: four per hole, four holes (built programmatically). -----
    # hole 0
    h0_north, h0_south, h0_east, h0_west = (
        _rim_walls_for_hole("Hole0", *HOLE_CENTERS_XY[0], HOLE_RADII[0]).values()
    )
    # hole 1
    h1_north, h1_south, h1_east, h1_west = (
        _rim_walls_for_hole("Hole1", *HOLE_CENTERS_XY[1], HOLE_RADII[1]).values()
    )
    # hole 2
    h2_north, h2_south, h2_east, h2_west = (
        _rim_walls_for_hole("Hole2", *HOLE_CENTERS_XY[2], HOLE_RADII[2]).values()
    )
    # hole 3
    h3_north, h3_south, h3_east, h3_west = (
        _rim_walls_for_hole("Hole3", *HOLE_CENTERS_XY[3], HOLE_RADII[3]).values()
    )

    # ----- Franka robot (OSC actuator/gravity overrides, unchanged). -----
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0
    robot.spawn.rigid_props.disable_gravity = False
    robot.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=PEG_FRICTION_STATIC,
        dynamic_friction=PEG_FRICTION_DYNAMIC,
        restitution=PEG_RESTITUTION,
    )
    robot.init_state.joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.3,
        "panda_joint3": 0.0,
        "panda_joint4": -2.410,
        "panda_joint5": 0.0,
        "panda_joint6": 3.037,
        "panda_joint7": 0.741,
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    }

    # ----- Four pegs (identical, different pick locations). -----
    peg0 = _peg_cfg(0, PEG_PLACE_POSITIONS[0])
    peg1 = _peg_cfg(1, PEG_PLACE_POSITIONS[1])
    peg2 = _peg_cfg(2, PEG_PLACE_POSITIONS[2])
    peg3 = _peg_cfg(3, PEG_PLACE_POSITIONS[3])
