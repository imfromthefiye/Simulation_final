import math
import random
from typing import TYPE_CHECKING, Optional

import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

try:
    import Box2D
    from Box2D.b2 import (
        b2World,
        contactListener,
        fixtureDef,
        polygonShape,
        circleShape,
        prismaticJointDef,
    )
except ImportError:
    raise DependencyNotInstalled(
        "Box2D is not installed, run `pip install gymnasium[box2d]`"
    )

# ================== Environment Constants ==================
FPS = 60
TIME_STEP = 1.0 / 60.0
VEL_ITERS, POS_ITERS = 100, 3

# Physical dimensions in meters
CHAMBER_WIDTH = 0.80
CHAMBER_HEIGHT = 1.00
MAX_PENETRATION = 0.80

# Rendering scale (px per meter)
SCALE = 500.0
VIEWPORT_W = int(CHAMBER_WIDTH * SCALE)
VIEWPORT_H = int(CHAMBER_HEIGHT * SCALE)

# Soil/grain parameters
D50 = 0.015           # mean grain size (m)
D_MIN, D_MAX = 0.01, 0.02  # range around D50
TARGET_FILL_HEIGHT = 0.20  # shallow deposit depth (m)
MU_DROP = 0.4
SETTLE_GRAVITY = 0.1
NORMAL_GRAVITY = 9.81
DAMPING = 0.5
MAX_PARTICLES = 500

# Probe parameters
D_PROBE = 0.05       # probe diameter (m)
TIP_LENGTH = 0.22    # tip length (m)
SHAFT_LENGTH = 0.50  # shaft length (m)
EXP_RATIO = 1.2      # shaft expansion ratio
PRISMATIC_LIMIT = 0.10  # max tip travel (m)

if TYPE_CHECKING:
    import Box2D.b2

# ================== Contact Detection ==================
class ContactDetector(contactListener):
    """
    Capture probe–soil contacts for force monitoring.
    """
    def __init__(self, env):
        super().__init__()
        self.env = env

    def BeginContact(self, contact):
        pass

    def EndContact(self, contact):
        pass

# ================== Helper Functions ==================
def create_chamber(world,
                   width=CHAMBER_WIDTH,
                   height=CHAMBER_HEIGHT,
                   wall_thickness=0.01,
                   wall_friction=0.05,
                   floor_friction=0.0):
    """
    Build static chamber: two vertical walls and a floor.
    """
    body = world.CreateStaticBody()
    half_w = width / 2.0

    # Left wall
    body.CreateFixture(
        shape=polygonShape(
            box=(wall_thickness/2, height/2),
            center=(-half_w + wall_thickness/2, height/2)
        ),
        friction=wall_friction
    )
    # Right wall
    body.CreateFixture(
        shape=polygonShape(
            box=(wall_thickness/2, height/2),
            center=(half_w - wall_thickness/2, height/2)
        ),
        friction=wall_friction
    )
    # Floor
    body.CreateFixture(
        shape=polygonShape(
            box=(width/2, wall_thickness/2),
            center=(0.0, wall_thickness/2)
        ),
        friction=floor_friction
    )
    return body


def rain_grains(world,
                inlet_x=0.0,
                inlet_y=None,
                delta_x=0.10,
                d_min=D_MIN,
                d_max=D_MAX,
                target_fill_height=TARGET_FILL_HEIGHT,
                mu_drop=MU_DROP,
                settling_gravity=SETTLE_GRAVITY,
                normal_gravity=NORMAL_GRAVITY,
                damping=DAMPING,
                max_particles=MAX_PARTICLES):
    """
    Deposit loose, shallow sand by raining grains until fill height or max count.
    """
    if inlet_y is None:
        inlet_y = height = CHAMBER_HEIGHT + 0.05

    world.gravity = (0.0, -settling_gravity)
    grains = []

    # Rain grains
    while True:
        max_y = max((g.position.y for g in grains), default=0.0)
        if max_y >= target_fill_height or len(grains) >= max_particles:
            break

        x = inlet_x + random.uniform(-delta_x, delta_x)
        r = random.uniform(d_min/2.0, d_max/2.0)
        grain = world.CreateDynamicBody(
            position=(x, inlet_y),
            fixtures=fixtureDef(
                shape=circleShape(radius=r),
                density=1.0,
                friction=mu_drop
            ),
            linearDamping=damping
        )
        grains.append(grain)
        world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)

    # Settle grains
    threshold = 0.01
    while True:
        world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
        max_vel = max(math.hypot(g.linearVelocity.x, g.linearVelocity.y) for g in grains)
        if max_vel < threshold:
            break

    # Reset gravity and remove damping
    world.gravity = (0.0, -normal_gravity)
    for g in grains:
        g.linearDamping = 0.0

    return grains


def create_probe(world,
                 d_probe=D_PROBE,
                 tip_length=TIP_LENGTH,
                 shaft_length=SHAFT_LENGTH,
                 expansion_ratio=EXP_RATIO,
                 prismatic_limit=PRISMATIC_LIMIT):
    """
    Build bio-inspired probe: shaft + tip on prismatic joint.
    """
    # Shaft
    shaft = world.CreateDynamicBody(
        position=(0.0, CHAMBER_HEIGHT + shaft_length/2),
        fixtures=fixtureDef(
            shape=polygonShape(box=(d_probe/2, shaft_length/2)),
            density=1.0,
            friction=0.2
        )
    )

    # Tip
    tip = world.CreateDynamicBody(
        position=(0.0, CHAMBER_HEIGHT + shaft_length + tip_length/2),
        fixtures=fixtureDef(
            shape=polygonShape(box=(d_probe/4, tip_length/2)),
            density=1.0,
            friction=0.2
        )
    )

    joint_def = prismaticJointDef(
        bodyA=shaft,
        bodyB=tip,
        localAxisA=(0, 1),
        enableLimit=True,
        lowerTranslation=0.0,
        upperTranslation=prismatic_limit,
        enableMotor=False,
    )
    tip_joint = world.CreateJoint(joint_def)
    return shaft, tip, tip_joint


def extend_tip(tip_joint,
               distance=PRISMATIC_LIMIT,
               speed=0.02):
    """
    Extend tip to `distance` along prismatic joint at `speed`.
    """
    tip_joint.enableMotor = True
    tip_joint.motorSpeed = speed
    tip_joint.upperTranslation = distance


def retract_tip(tip_joint,
                speed=0.02):
    """
    Retract tip to zero translation at `speed`.
    """
    tip_joint.enableMotor = True
    tip_joint.motorSpeed = -speed
    tip_joint.lowerTranslation = 0.0


def advance_body(shaft,
                 tip,
                 target_depth=MAX_PENETRATION,
                 force_limit=None):
    """
    Drive entire probe downward until tip_y ≥ target_depth or force ≥ force_limit.
    """
    # TODO: implement kinematic lowering with force check
    pass


def monitor_forces(probe_joint):
    """
    Return current reaction force on prismatic joint (tip).
    """
    fx, fy = probe_joint.GetReactionForce(TIME_STEP)
    return math.hypot(fx, fy)


def draw_all(screen,
             chamber_body,
             grains,
             shaft,
             tip):
    """
    Render chamber, grains, and probe via Pygame.
    """
    screen.fill((255, 255, 255))
    # Grains
    for g in grains:
        x = int(g.position.x * SCALE)
        y = VIEWPORT_H - int(g.position.y * SCALE)
        r = int(g.fixtures[0].shape.radius * SCALE)
        pygame.draw.circle(screen, (200, 200, 200), (x, y), r)
    # Shaft & tip
    for body, color in [(shaft, (100, 100, 255)), (tip, (255, 100, 100))]:
        for f in body.fixtures:
            verts = [(body.transform * v) * SCALE for v in f.shape.vertices]
            pts = [(int(x), VIEWPORT_H - int(y)) for x, y in verts]
            pygame.draw.polygon(screen, color, pts)
    pygame.display.flip()


def run_simulation():
    """
    Test script: build world, rain grains, build probe, step and render.
    """
    pygame.init()
    screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
    clock = pygame.time.Clock()

    world = b2World(gravity=(0.0, -SETTLE_GRAVITY))
    world.contactListener = ContactDetector(None)

    chamber = create_chamber(world)
    grains = rain_grains(world)
    shaft, tip, tip_joint = create_probe(world)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
        draw_all(screen, chamber, grains, shaft, tip)
        clock.tick(FPS)
    pygame.quit()
