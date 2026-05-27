"""Scene configuration for the peg-in-hole insertion task.

This module defines the shared world: a Franka Panda on a stand, a fixed
socket in front of it, and a free-floating peg (visual placeholder for pass-1).
The exact same `InsertionSceneCfg` is consumed by:

  - pass-1 sim test (static hold above socket)
  - pass-2 DMP rollout
  - future diffusion-policy data generation
  - future residual-RL training env

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
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # noqa: I001 — must come after AppLauncher


# ─── Geometry constants. Single source of truth for downstream consumers. ────
PEG_RADIUS = 0.008
PEG_LENGTH = 0.05
SOCKET_DEPTH = 0.04
SOCKET_WIDTH = 0.05

SOCKET_X = 0.4
SOCKET_Y = 0.3
SOCKET_BASE_Z = 0.36
SOCKET_TOP_Z = SOCKET_BASE_Z + SOCKET_DEPTH

SOCKET_BASE_POS = (SOCKET_X, SOCKET_Y, SOCKET_BASE_Z)


@configclass
class InsertionSceneCfg(InteractiveSceneCfg):
    """Franka + peg + socket on a tabletop."""

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

    # ----- Tabletop mount (raises the Franka above the floor) -----
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
        ),
    )

    # ----- Socket (static visual landmark for pass-1) -----
    socket = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Socket",
        spawn=sim_utils.CuboidCfg(
            size=(SOCKET_WIDTH, SOCKET_WIDTH, SOCKET_DEPTH),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.2, 0.6),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            activate_contact_sensors=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(SOCKET_X, SOCKET_Y, SOCKET_BASE_Z + SOCKET_DEPTH / 2),
        ),
    )

    # ----- Peg (free rigid body, not yet attached in pass-1) -----
    peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        spawn=sim_utils.CylinderCfg(
            radius=PEG_RADIUS,
            height=PEG_LENGTH,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.6, 0.2)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(SOCKET_X, SOCKET_Y, 0.7),
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
