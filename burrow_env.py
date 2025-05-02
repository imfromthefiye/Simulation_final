import pygame
from Box2D import (
    b2World, b2PolygonShape, b2FixtureDef, b2CircleShape,
    b2PrismaticJointDef, b2ContactListener, b2ContactImpulse
)

# ================== Environment Constants ==================
CHAMBER_WIDTH = 0.8   # meters
CHAMBER_HEIGHT = 1.0  # meters
MARGIN = 0.1          # meters border
SCALE = 400.0         # px per meter
VIEWPORT_W = int((CHAMBER_WIDTH + 2 * MARGIN) * SCALE)
VIEWPORT_H = int((CHAMBER_HEIGHT + 2 * MARGIN) * SCALE)
FPS = 60
TIME_STEP = 1.0 / FPS
VEL_ITERS, POS_ITERS = 10, 10

# Grain parameters
GRAIN_RADIUS = 0.015  # m

# ================== World Setup ==================
def create_grains(world):
    grains = []
    
    # Define the grain arrangement first
    grain_diameter = GRAIN_RADIUS * 2
    
    # Calculate rows and columns based on chamber width, not including walls yet
    COLS = int(CHAMBER_WIDTH / grain_diameter)
    
    # Fill 1/3 of chamber height with grains
    soil_height = CHAMBER_HEIGHT / 3
    ROWS = int(soil_height / (grain_diameter * 0.866))  # 0.866 = sqrt(3)/2
    
    # Center the array horizontally
    x0 = -(COLS-1) * grain_diameter / 2
    
    # Position grains starting at y=0 (we'll adjust the chamber floor later)
    y0 = GRAIN_RADIUS  # Bottom of first grain at y=0
    
    for i in range(ROWS):
        for j in range(COLS):
            # Offset alternate rows for hexagonal packing
            x = x0 + j * grain_diameter + (0.5 * grain_diameter if i % 2 else 0)
            y = y0 + i * (grain_diameter * 0.866)  # sqrt(3)/2 ≈ 0.866
            
            # Skip grains that would extend beyond desired width
            if abs(x) > (CHAMBER_WIDTH/2 - GRAIN_RADIUS):
                continue
                
            body = world.CreateDynamicBody(
                position=(x, y),
                fixtures=b2FixtureDef(
                    shape=b2CircleShape(radius=GRAIN_RADIUS),
                    density=0.8,
                    friction=0.3,
                    restitution=0.1
                ),
                linearDamping=0.8
            )
            body.gravityScale = 0.0
            grains.append(body)
    
    return grains

def create_chamber(world):
    # Wall thickness
    t = 0.01
    
    # Calculate half width
    half_w = CHAMBER_WIDTH / 2
    
    walls = []
    
    # Left wall - place it just outside the leftmost grains
    walls.append(world.CreateStaticBody(
        position=(-half_w - t/2, CHAMBER_HEIGHT/2),
        fixtures=b2FixtureDef(shape=b2PolygonShape(box=(t/2, CHAMBER_HEIGHT/2)), friction=0.3)
    ))
    
    # Right wall - place it just outside the rightmost grains
    walls.append(world.CreateStaticBody(
        position=(half_w + t/2, CHAMBER_HEIGHT/2),
        fixtures=b2FixtureDef(shape=b2PolygonShape(box=(t/2, CHAMBER_HEIGHT/2)), friction=0.3)
    ))
    
    # Floor - place it directly beneath the grains at y=0
    walls.append(world.CreateStaticBody(
        position=(0.0, -t/2),
        fixtures=b2FixtureDef(shape=b2PolygonShape(box=(half_w + t, t/2)), friction=0.5)
    ))
    
    return walls

# ================== Probe Parameters ==================
# Probe parameters
SHAFT_W, SHAFT_H = 0.05, 0.5
TIP_BASE = SHAFT_W
TIP_HEIGHT = SHAFT_W * 0.6

# Position the probe closer to the soil bed
soil_height = CHAMBER_HEIGHT / 3
PROBE_POS_Y = soil_height + 0.05 + SHAFT_H/2  # Only small clearance above soil
EXPANSION_RATIO = 1.2
TIP_EXTENSION_SPEED = 0.03

# Control mappings
KEY_EXPAND_SHAFT = pygame.K_e
KEY_CONTRACT_SHAFT = pygame.K_c
KEY_EXTEND_TIP = pygame.K_t
KEY_MOVE_DOWN = pygame.K_m

# ================== Helper: world->screen ==================
def world_to_screen(x, y):
    # Move the origin to bottom-left corner of the chamber
    # This positions the chamber floor at the bottom of the viewport with a small margin
    sx = int((x + CHAMBER_WIDTH/2 + MARGIN) * SCALE)
    
    # Flip Y-axis and position the origin at the bottom with a small margin
    # This ensures the floor appears at the bottom of the screen
    bottom_margin = MARGIN * 2  # Extra margin at the bottom
    sy = int(VIEWPORT_H - ((y + bottom_margin) * SCALE))
    
    return sx, sy

def world_to_screen_pts(pt):
    x, y = pt
    return world_to_screen(x, y)

# ================== Contact Listener ==================
class ContactDetector(b2ContactListener):
    def __init__(self, shaft_body, tip_body):
        super().__init__()
        self.shaft = shaft_body
        self.tip = tip_body
        self.shaft_impulse = 0.0
        self.tip_impulse = 0.0

    def PostSolve(self, contact, impulse: b2ContactImpulse):
        total = sum(impulse.normalImpulses)
        bodies = (contact.fixtureA.body, contact.fixtureB.body)
        if self.shaft in bodies:
            self.shaft_impulse += total
        if self.tip in bodies:
            self.tip_impulse += total

# ================== World Setup ==================
def create_probe(world):
    shaft = world.CreateDynamicBody(
        position=(0.0, PROBE_POS_Y),
        fixtures=b2FixtureDef(
            shape=b2PolygonShape(box=(SHAFT_W/2, SHAFT_H/2)),
            density=1.0, friction=0.2
        )
    )
    shaft.gravityScale = 0.0

    verts = [
        (-TIP_BASE/2, -SHAFT_H/2),
        ( TIP_BASE/2, -SHAFT_H/2),
        (   0.0    , -SHAFT_H/2 - TIP_HEIGHT)
    ]
    tip = world.CreateDynamicBody(
        position=(0.0, PROBE_POS_Y),
        fixtures=b2FixtureDef(
            shape=b2PolygonShape(vertices=verts),
            density=1.0, friction=0.2
        )
    )
    tip.gravityScale = 0.0

    joint_def = b2PrismaticJointDef()
    anchor = (0.0, PROBE_POS_Y - SHAFT_H/2)
    joint_def.Initialize(shaft, tip, anchor=anchor, axis=(0, -1))
    joint_def.enableLimit = True
    joint_def.lowerTranslation = 0.0
    joint_def.upperTranslation = TIP_HEIGHT
    joint_def.enableMotor = True
    joint_def.motorSpeed = -TIP_EXTENSION_SPEED
    joint_def.maxMotorForce = 500.0
    joint = world.CreateJoint(joint_def)
    return shaft, tip, joint

# ================== Probe Actions ==================
def move_body_down(shaft, tip, joint):
    joint.enableMotor = False
    d = joint.translation
    shaft.position = (shaft.position.x, shaft.position.y - d)
    tip.position = (tip.position.x, tip.position.y)
    print(f"Moving shaft down by {d}m to close gap")
    print(f"Joint translation after move: {joint.translation}")

# ================ Rendering =================
def draw_probe(screen, probe):
    shaft, tip, _ = probe
    for f in shaft.fixtures:
        pts = [world_to_screen_pts(shaft.transform * v) for v in f.shape.vertices]
        pygame.draw.polygon(screen, (100, 100, 255), pts)
    for f in tip.fixtures:
        pts = [world_to_screen_pts(tip.transform * v) for v in f.shape.vertices]
        pygame.draw.polygon(screen, (255, 100, 100), pts)

# ================ CPT Test ==================
def run_cpt_test():
    pygame.init()
    screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
    pygame.display.set_caption("CPT Test")
    clock = pygame.time.Clock()

    world = b2World(gravity=(0.0, 0.0))
    create_chamber(world)
    create_grains(world)
    shaft, tip, joint = create_probe(world)
    detector = ContactDetector(shaft, tip)
    world.contactListener = detector

    pygame.font.init()
    font = pygame.font.Font(None, 24)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == KEY_EXTEND_TIP:
                    joint.enableMotor = True
                    joint.motorSpeed = -TIP_EXTENSION_SPEED  # Negative for downward movement
                    joint.maxMotorForce = 500.0
            elif e.type == pygame.KEYUP:
                if e.key == KEY_EXTEND_TIP:
                    joint.enableMotor = False

        # Substepping CPT test
        world.Step(TIME_STEP / 4, VEL_ITERS, POS_ITERS)

        screen.fill((255, 255, 255))
        for body in world.bodies:
            for f in body.fixtures:
                shape = f.shape
                if isinstance(shape, b2CircleShape):
                    x, y = world_to_screen(body.position.x, body.position.y)
                    pygame.draw.circle(screen, (150, 150, 150), (x, y), int(GRAIN_RADIUS * SCALE))
                elif isinstance(shape, b2PolygonShape):
                    pts = [world_to_screen_pts(body.transform * v) for v in shape.vertices]
                    pygame.draw.polygon(screen, (50, 50, 50), pts)

        draw_probe(screen, (shaft, tip, joint))

        tip_ext = joint.translation
        text = font.render(f"Tip Extension: {tip_ext:.3f}/{TIP_HEIGHT:.3f}m", True, (0, 0, 0))
        screen.blit(text, (10, 10))
        force_text = font.render(f"Tip Force: {detector.tip_impulse:.1f} N·s", True, (0, 0, 0))
        screen.blit(force_text, (10, 40))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    run_cpt_test()
