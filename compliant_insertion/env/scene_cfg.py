"""Scene configuration for the peg-in-hole insertion task.

This module defines the shared world: a Franka Panda on a stand, a table in
front of it with a socket block that has a real cylindrical hole, and a peg
that is rigidly attached to the Franka's hand (i.e. "pre-grasped"). The same
`InsertionSceneCfg` is consumed by:

  - pass-1 sim test (static hold above socket)
  - pass-2 DMP rollout (peg-tip-frame trajectories, converted to hand-frame
    via PEG_TIP_OFFSET_HAND — see below)
  - future diffusion-policy data generation
  - future residual-RL training env

Peg attachment (kinematic tracking model)
-----------------------------------------
The peg is spawned as a `RigidObjectCfg` at the world level with gravity
disabled, and the run loop kinematically writes its pose from the hand's
pose every physics step (see `peg_pose_from_hand()` below for the math).
This is the pattern used by Isaac Lab's official Factory insertion tasks
and is robust across Isaac Sim versions, where spawning collision prims
under articulation links has version-dependent behaviour.

Mechanically this skips the grasp-controller phase, which for DMP rollout
and downstream learning is exactly the simplification we want: the
relevant phase begins *after* the peg is in hand. Swap to a runtime grasp
constraint later when you want to model grasp variation for the diffusion
dataset.

Frame convention for the offset
-------------------------------
`panda_hand` has +Z pointing out of the gripper along the approach axis (the
direction the fingers point), with the origin between the finger bases. The
peg is mounted with its cylindrical axis aligned to `panda_hand` +Z:

    panda_hand origin ─── +Z ───►
        │
        ├── PEG_BASE_OFFSET_Z  (peg base sits at the fingertip plane)
        │     │
        │     │ cylinder body (length = PEG_LENGTH)
        │     ▼
        └── PEG_TIP_OFFSET_Z   (peg tip — what we actually want to command)

In the hand frame the tip offset is simply `(0, 0, PEG_TIP_OFFSET_Z)`. To
convert a peg-tip target in the *robot base frame* into a hand target in the
*robot base frame*, rotate the hand-frame offset by the commanded hand
orientation and subtract:

    hand_target_pos_b = peg_tip_target_pos_b - quat_rotate(hand_quat_b,
                                                           PEG_TIP_OFFSET_HAND)

When orientation is held constant at home_quat_b (the current DMP rollout
case), you can precompute the rotated offset once and reuse it every step.

Two changes from a vanilla high-PD Franka, both copied directly from the
official OSC tutorial (scripts/tutorials/05_controllers/run_osc.py):

  1. Arm actuator stiffness and damping are zeroed at the CONFIG level. With
     OSC the joint torques come from the controller, not from an internal
     PD loop — leaving the high-PD gains in place means the actuator fights
     OSC's commands and the EE tracks slowly or oscillates.
  2. Robot gravity is disabled. The OSC config below uses
     `gravity_compensation=False` (the tutorial default), so without this
     flag the arm would sag. Toggle both together if you ever want real
     gravity in the loop.

Hole geometry
-------------
Isaac Sim's primitive `CuboidCfg`/`CylinderCfg` don't support boolean
subtraction, so the hole is built as a rim of four cuboid walls around an
empty cylindrical region. This keeps clean contact geometry (no concave
mesh tricks, no CSG at load time) and exposes a single `HOLE_CLEARANCE`
parameter for tolerance studies — change it, restart the sim, done.

The rim sits on top of a flat socket base block so the peg can't fall
through. Wall thickness, hole depth, and the surrounding block footprint
are all configurable below.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # noqa: I001 — must come after AppLauncher


# ─── Peg geometry. ───────────────────────────────────────────────────────────
PEG_RADIUS = 0.008
PEG_LENGTH = 0.05

# ─── Hole / socket geometry. ─────────────────────────────────────────────────
# Tolerance knob: hole_radius = peg_radius + HOLE_CLEARANCE.
# Typical industrial values:
#   loose fit       (H11/c11) → ~1.0 mm clearance on a 16 mm peg
#   sliding fit     (H8/f7)   → ~0.05 mm
#   close-running   (H7/g6)   → ~0.02 mm
# For sim, anything below ~0.5 mm gets numerically nasty without sub-stepping
# and contact tuning. Start generous, tighten once the DMP rollout is clean.
HOLE_CLEARANCE = 0.003        # 3 mm — generous, easy for the DMP to enter
HOLE_RADIUS = PEG_RADIUS + HOLE_CLEARANCE
HOLE_DEPTH = 0.04             # how deep the peg can sink before bottoming out

# The socket block: a flat slab the hole-rim sits on. Its top surface is the
# "table-top of the fixture", what the peg's tip will eventually contact if
# you misaim. Make it visibly larger than the hole so the visual is clear.
SOCKET_BLOCK_SIZE_XY = 0.10   # 10 cm × 10 cm footprint
SOCKET_BLOCK_HEIGHT = 0.01    # 1 cm thick base under the rim

# Rim walls — four cuboids surrounding the hole. Wall thickness controls how
# robust the "lip" of the hole is; the outer footprint matches the socket
# block so the assembly looks like one fixture.
RIM_WALL_THICKNESS = (SOCKET_BLOCK_SIZE_XY - 2 * HOLE_RADIUS) / 2  # auto-fit
RIM_HEIGHT = HOLE_DEPTH

# ─── Placement. ──────────────────────────────────────────────────────────────
# Where the fixture sits on the table mount. The Z values below assume the
# `Stand` USD puts its top surface roughly at z=0; if you swap the mount,
# re-measure and adjust SOCKET_BASE_Z.
SOCKET_X = 0.4
SOCKET_Y = 0.3
SOCKET_BASE_Z = 0.36          # top of the table mount

# Layered Z stack:
#   [base_z, base_z + block_height]   → solid socket block
#   [base_z + block_height, ...+rim]  → rim walls with hole
SOCKET_BLOCK_TOP_Z = SOCKET_BASE_Z + SOCKET_BLOCK_HEIGHT
SOCKET_TOP_Z = SOCKET_BLOCK_TOP_Z + RIM_HEIGHT   # top of the rim — the
                                                 # surface a misaimed peg
                                                 # tip would hit

SOCKET_BASE_POS = (SOCKET_X, SOCKET_Y, SOCKET_BASE_Z)

# Hole-center pose (peg tip target for a successful insertion, at rim top).
HOLE_CENTER_TOP = (SOCKET_X, SOCKET_Y, SOCKET_TOP_Z)
HOLE_CENTER_BOTTOM = (SOCKET_X, SOCKET_Y, SOCKET_BLOCK_TOP_Z)

# ─── Peg-in-gripper attachment ───────────────────────────────────────────────
# The peg is a normal world-level RigidObjectCfg whose pose is kinematically
# driven from `panda_hand` every physics step. Its cylindrical axis is
# defined to be aligned with panda_hand +Z. The constants below describe
# where along that +Z the peg sits relative to the hand origin.
#
# PANDA_FINGERTIP_OFFSET_Z is the well-known TCP offset for the default Franka
# hand: the distance from the `panda_hand` link origin to the tip of the
# closed fingers. The peg's base sits at the fingertip plane (i.e. the peg
# "starts where the fingers end") so its body extends from there along +Z.
PANDA_FINGERTIP_OFFSET_Z = 0.1034  # meters; standard Franka panda_hand → TCP

# Peg base (the end held by the gripper) and tip (the end that enters the hole)
# expressed as offsets in the panda_hand frame.
PEG_BASE_OFFSET_Z = PANDA_FINGERTIP_OFFSET_Z
PEG_TIP_OFFSET_Z = PEG_BASE_OFFSET_Z + PEG_LENGTH

# Hand-frame offset vector for the peg tip. Use this to convert peg-tip
# targets to hand targets in downstream code:
#
#     from isaaclab.utils.math import quat_apply
#     tip_offset_b = quat_apply(hand_quat_b, PEG_TIP_OFFSET_HAND_TENSOR)
#     hand_target_pos_b = peg_tip_target_pos_b - tip_offset_b
#
# When orientation is fixed at home_quat_b, precompute tip_offset_b once.
PEG_TIP_OFFSET_HAND = (0.0, 0.0, PEG_TIP_OFFSET_Z)
PEG_BASE_OFFSET_HAND = (0.0, 0.0, PEG_BASE_OFFSET_Z)

# Peg body center in the hand frame (cylinder spawn pose). The cylinder
# primitive is centered on its origin, so to place its base at PEG_BASE_OFFSET_Z
# we shift the center up by half the length.
PEG_BODY_CENTER_HAND_Z = PEG_BASE_OFFSET_Z + PEG_LENGTH / 2

# ─── Grasp / friction parameters ──────────────────────────────────────────────
# The peg is a DYNAMIC body held by friction. These coefficients are high on
# purpose: a 50 g cylinder gripped by two small fingertip patches will slip at
# default (~0.5) friction the moment insertion reaction load arrives. Values
# around 1.0–1.5 are typical for "rubberized fingertip on metal peg" in sim and
# keep the grasp stable through insertion. Raise PEG_FRICTION if you still see
# slip; lower it if the peg sticks unrealistically.
PEG_FRICTION_STATIC = 1.2
PEG_FRICTION_DYNAMIC = 1.0
PEG_RESTITUTION = 0.0  # no bounce — dead contacts settle faster


def peg_grasp_pose_from_hand(hand_pos_w, hand_quat_w):
    """World pose to PLACE the dynamic peg for a clean pinch grasp.

    Same geometry as peg_pose_from_hand (body center at PEG_BODY_CENTER_HAND_Z
    along the hand +Z), but named separately because its role is different:
    this is used ONCE at grasp time (and on each reset's held-state write) to
    position the peg centered between the fingers, NOT to drive it every step.
    After the grasp is established, the peg moves under contact dynamics alone.
    """
    import torch
    from isaaclab.utils.math import quat_apply

    offset_hand = torch.tensor(
        [0.0, 0.0, PEG_BODY_CENTER_HAND_Z],
        device=hand_pos_w.device,
        dtype=hand_pos_w.dtype,
    ).expand_as(hand_pos_w)
    offset_w = quat_apply(hand_quat_w, offset_hand)
    return hand_pos_w + offset_w, hand_quat_w


def peg_tip_from_body(peg_pos_w, peg_quat_w):
    """Tip position from the DYNAMIC peg's own body pose.

    Use this (not peg_tip_pose_from_hand) for the success check once the peg
    is grasped: it reflects where the tip ACTUALLY is, including any grasp
    compliance / slip / wobble. The tip is +PEG_LENGTH/2 along the peg's
    local +Z from its body center.
    """
    import torch
    from isaaclab.utils.math import quat_apply

    half_len = torch.tensor(
        [0.0, 0.0, PEG_LENGTH / 2],
        device=peg_pos_w.device,
        dtype=peg_pos_w.dtype,
    ).expand_as(peg_pos_w)
    return peg_pos_w + quat_apply(peg_quat_w, half_len)

# ─── Rim wall poses ──────────────────────────────────────────────────────────
# Four walls, axis-aligned, surrounding a square that inscribes the hole.
# Each wall is a thin slab; together they leave a square void of side
# (2 * HOLE_RADIUS), which is the bounding square of the cylindrical hole.
# The peg is round so the corners of this square are "wasted" — that's fine
# for a first pass; swap to an octagonal rim later if you want a tighter
# visual match to a real round hole.
_WALL_Z = SOCKET_BLOCK_TOP_Z + RIM_HEIGHT / 2  # vertical center of the rim
_OUTER_HALF = SOCKET_BLOCK_SIZE_XY / 2
_INNER_HALF = HOLE_RADIUS
_WALL_OFFSET = (_OUTER_HALF + _INNER_HALF) / 2  # center of each wall slab

# Wall sizes: (size_x, size_y, size_z)
_WALL_SIZE_LONG = (SOCKET_BLOCK_SIZE_XY, RIM_WALL_THICKNESS, RIM_HEIGHT)  # north/south
_WALL_SIZE_SHORT = (RIM_WALL_THICKNESS, 2 * _INNER_HALF, RIM_HEIGHT)      # east/west


# ─── Pose helper: drive the peg from the hand ─────────────────────────────────
def peg_pose_from_hand(hand_pos_w, hand_quat_w):
    """Compute the peg's world pose given the panda_hand's world pose.

    The peg's body-center sits at PEG_BODY_CENTER_HAND_Z along the hand's
    +Z axis, with no rotation relative to the hand. So the world-frame peg
    pose is:

        peg_pos_w  = hand_pos_w + R(hand_quat_w) @ (0, 0, PEG_BODY_CENTER_HAND_Z)
        peg_quat_w = hand_quat_w

    The function is shape-polymorphic and works on torch tensors of shape
    [N, 3] / [N, 4] (typical Isaac Lab batched-env case) thanks to
    `quat_apply` broadcasting.

    Import this in run_dmp_in_sim.py and call it once per step (or only on
    reset if the hand orientation is constant — but per-step is cheap).
    """
    import torch
    from isaaclab.utils.math import quat_apply

    offset_hand = torch.tensor(
        [0.0, 0.0, PEG_BODY_CENTER_HAND_Z],
        device=hand_pos_w.device,
        dtype=hand_pos_w.dtype,
    ).expand_as(hand_pos_w)
    offset_w = quat_apply(hand_quat_w, offset_hand)
    peg_pos_w = hand_pos_w + offset_w
    peg_quat_w = hand_quat_w
    return peg_pos_w, peg_quat_w


def peg_tip_pose_from_hand(hand_pos_w, hand_quat_w):
    """World-frame position of the peg tip (the working end).

    Same math as peg_pose_from_hand but with PEG_TIP_OFFSET_Z instead of
    the body-center offset. Use this for trajectory tracking metrics and
    for the insertion success check below.
    """
    import torch
    from isaaclab.utils.math import quat_apply

    offset_hand = torch.tensor(
        [0.0, 0.0, PEG_TIP_OFFSET_Z],
        device=hand_pos_w.device,
        dtype=hand_pos_w.dtype,
    ).expand_as(hand_pos_w)
    tip_offset_w = quat_apply(hand_quat_w, offset_hand)
    return hand_pos_w + tip_offset_w


# ─── Geometric success criterion ──────────────────────────────────────────────
def insertion_success(
    tip_pos_w,
    hole_center_w=(SOCKET_X, SOCKET_Y, SOCKET_BLOCK_TOP_Z),
    hole_radius=HOLE_RADIUS,
    rim_top_z=SOCKET_TOP_Z,
    depth_threshold=0.5,
):
    """Geometric peg-in-hole success check.

    Returns a `(success, lateral_err, depth_frac)` tuple where:

      - `success` is a bool tensor [N] — True iff the tip is laterally
         inside the hole AND has descended past `depth_threshold` of the
         rim height below the rim top.
      - `lateral_err` is the XY distance from the tip to the hole axis
        in meters [N] — useful as an RL reward shaping term or as a
        progress signal during DMP rollout debugging.
      - `depth_frac` is how far below the rim top the tip has descended,
        normalized by rim height [N]. 0.0 == at rim top, 1.0 == fully
        seated at the block top. Negative means the tip is above the rim.

    The check is **geometric only** — it ignores whether the peg
    physically would have collided with the rim on the way in. For a
    kinematic peg this is exactly what we want: a clean signal of "the
    DMP commanded a trajectory that ends inside the hole" independent
    of contact dynamics.

    For a stricter check (peg never violated the rim walls during
    descent), accumulate `lateral_err <= hole_radius` over all steps
    after the tip passed below `rim_top_z` — if any step had a larger
    lateral error, the rollout would have collided in reality.

    Args:
        tip_pos_w: [N, 3] tip positions in world frame.
        hole_center_w: 3-tuple, the (x, y, z) of the hole's BOTTOM-CENTER
            (where the tip should land when fully seated).
        hole_radius: scalar, the cylindrical clearance radius.
        rim_top_z: scalar, the world Z of the rim's top surface.
        depth_threshold: scalar in [0, 1], the fraction of rim height
            the tip must descend below `rim_top_z` to count as
            successful. Default 0.5 (half the hole depth).
    """
    import torch

    hole_center = torch.tensor(
        hole_center_w, device=tip_pos_w.device, dtype=tip_pos_w.dtype
    )
    # Lateral (XY) distance from the hole axis.
    lateral_err = torch.norm(tip_pos_w[..., :2] - hole_center[:2], dim=-1)
    # How far below the rim top the tip has descended.
    rim_height = rim_top_z - hole_center_w[2]
    depth_below_rim = rim_top_z - tip_pos_w[..., 2]
    depth_frac = depth_below_rim / rim_height

    laterally_inside = lateral_err <= hole_radius
    deep_enough = depth_frac >= depth_threshold
    success = laterally_inside & deep_enough
    return success, lateral_err, depth_frac


@configclass
class InsertionSceneCfg(InteractiveSceneCfg):
    """Franka + socket-with-hole + peg on a tabletop."""

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

    # ----- Socket base block (the flat fixture the rim walls sit on) -----
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

    # ----- Rim wall: +Y (north of hole) -----
    rim_north = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RimNorth",
        spawn=sim_utils.CuboidCfg(
            size=_WALL_SIZE_LONG,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.6)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            activate_contact_sensors=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(SOCKET_X, SOCKET_Y + _WALL_OFFSET, _WALL_Z),
        ),
    )

    # ----- Rim wall: -Y (south of hole) -----
    rim_south = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RimSouth",
        spawn=sim_utils.CuboidCfg(
            size=_WALL_SIZE_LONG,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.6)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            activate_contact_sensors=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(SOCKET_X, SOCKET_Y - _WALL_OFFSET, _WALL_Z),
        ),
    )

    # ----- Rim wall: +X (east of hole) -----
    rim_east = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RimEast",
        spawn=sim_utils.CuboidCfg(
            size=_WALL_SIZE_SHORT,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.6)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            activate_contact_sensors=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(SOCKET_X + _WALL_OFFSET, SOCKET_Y, _WALL_Z),
        ),
    )

    # ----- Rim wall: -X (west of hole) -----
    rim_west = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RimWest",
        spawn=sim_utils.CuboidCfg(
            size=_WALL_SIZE_SHORT,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.6)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            activate_contact_sensors=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(SOCKET_X - _WALL_OFFSET, SOCKET_Y, _WALL_Z),
        ),
    )

    # ----- Franka robot -----
    # Mirror the tutorial's actuator/gravity overrides. These two blocks are
    # what make OSC work cleanly; do NOT zero stiffness in the run script.
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0
    robot.spawn.rigid_props.disable_gravity = True

    # Fingertip friction must be high too — the realized contact friction is
    # the combination of both materials, so a high-friction peg against
    # default fingers still slips. Match the peg's coefficients on the hand.
    robot.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=PEG_FRICTION_STATIC,
        dynamic_friction=PEG_FRICTION_DYNAMIC,
        restitution=PEG_RESTITUTION,
    )

    # ----- Initial joint configuration -----
    # Override the default Franka "ready" pose so the arm starts to the side
    # of the socket, not above it — this matches the spirit of a DMP rollout
    # where the start pose is "after grasping the peg, before approaching".
    #
    # These angles were tuned by hand against the current socket position
    # (SOCKET_X=0.4, SOCKET_Y=0.3). They put the hand roughly at
    # (x≈0.4, y≈0.1, z≈0.51) in base frame — i.e. 20 cm to the −Y side of
    # the socket, ~10 cm above the rim top — pointing down with the same
    # orientation as the default ready pose. Adjust if you reposition the
    # socket or want a different start.
    #
    # If you want a true Cartesian start specification ("hand at exactly
    # this XYZ"), the right next step is to run an IK solver once offline
    # (Isaac Lab's `DifferentialIKController` or `pinocchio`) and paste
    # the resulting joint angles here. For pass-2 the hand-tuned values
    # below are good enough; the DMP doesn't care that the start is
    # 5 mm off as long as it's reproducible.
    robot.init_state.joint_pos = {
        "panda_joint1": 0.6,     # rotate base toward +X+Y quadrant
        "panda_joint2": 0.3,     # shoulder pitch up (less squat than default)
        "panda_joint3": 0.0,
        "panda_joint4": -2.0,    # less elbow bend (was -2.81)
        "panda_joint5": 0.0,
        "panda_joint6": 2.3,     # wrist pitch to keep tip pointing down
        "panda_joint7": 0.741,   # default wrist roll (peg is symmetric)
        # Fingers — close them so the gripper "holds" the peg visually.
        "panda_finger_joint1": 0.0,
        "panda_finger_joint2": 0.0,
    }

    # ----- Peg (DYNAMIC rigid body, held by friction) -----
    # The peg is a normal dynamic body now. It is grasped once at startup and
    # the held-state is re-established on each reset (see the runner's reset
    # logic). From the moment of grasp onward, the peg's motion is determined
    # purely by contact: fingertip friction holding it, socket walls resisting
    # it during insertion. A jam or a slip therefore manifests as a shallow
    # final seating depth — which is exactly the (single) success signal we
    # measure on the peg's OWN body pose via peg_tip_from_body().
    #
    # High friction is essential: a 50 g peg on two small contact patches will
    # slip at default friction the instant insertion load arrives.
    peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
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
                # Dynamic: no kinematic flag, gravity ON. Solver iteration
                # counts bumped for stable two-patch grasp contacts.
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
            ),
        ),
        # Placeholder spawn; the runner writes the real grasp pose on the
        # first reset (peg centered between the fingers).
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
        ),
    )
