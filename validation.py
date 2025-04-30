# biaxial_test.py
import csv, signal, sys, math
import pygame
from Box2D.b2 import (world, dynamicBody, staticBody, kinematicBody,
                      circleShape, polygonShape)

# ——— Simulation parameters ———
DB        = 1.0      # disc diameter (m)
ROWS      = 8        # rows of hex packing
COLS      = 4        # discs per row
L0        = ROWS * DB * math.sqrt(3)/2  # initial specimen height
SIG2      = 1000.0   # confining stress σ₂ (N/m)
F2        = SIG2*DB  # total horizontal force per unit depth
P_SPEED   = 0.005    # top platen speed (m/s)
TIME_STEP = 1.0/60
VEL_IT    = 100
POS_IT    = 3

# ——— Logging setup ———
with open('log.csv','w', newline='') as f:
    csv.writer(f).writerow(('delta/L','sigma1/sigma2'))

def on_exit(sig, frame):
    print("\nExiting cleanly, log.csv saved")
    sys.exit(0)
signal.signal(signal.SIGINT, on_exit)

# ——— Build Box2D world ———
w = world(gravity=(0,0), doSleep=True)

# pack 32 discs in hex array, centred about x=0
bodies = []
x0 = -(COLS-1)*DB/2
y0 = -(ROWS-1)*(DB*math.sqrt(3)/2)/2
for i in range(ROWS):
    for j in range(COLS):
        x = x0 + j*DB + (0.5*DB if i%2 else 0)
        y = y0 + i*(DB*math.sqrt(3)/2)
        b = w.CreateDynamicBody(position=(x,y))
        b.CreateCircleFixture(radius=DB/2, density=1, friction=0.2)
        bodies.append(b)

miny = min(b.position.y for b in bodies) - DB/2
maxy = max(b.position.y for b in bodies) + DB/2

# bottom rigid platen
bottom = w.CreateStaticBody(
    position=(0, miny - 0.01),
    shapes=polygonShape(box=(COLS*DB/2, 0.01))
)

# top kinematic platen
top = w.CreateKinematicBody(
    position=(0, maxy + 0.01),
    shapes=polygonShape(box=(COLS*DB/2, 0.01))
)
top.linearVelocity = (0, -P_SPEED)

# ——— Pygame setup ———
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800
PPM = SCREEN_WIDTH / (COLS*DB*2)   # pixels per meter

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT), pygame.RESIZABLE)
clock  = pygame.time.Clock()

def to_screen(pos):
    x,y = pos
    return (
        int((x - x0 + COLS*DB/2)*PPM),
        int(SCREEN_HEIGHT - (y - (miny - 0.01))*PPM)
    )

# ——— Main loop ———
while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            on_exit(None,None)

    # apply horizontal confinement F2 to boundary particles
    for b in bodies:
        # find column index
        j = int(round((b.position.x - x0)/DB))
        if j == 0 or j == COLS-1:
            # full F2 on all side discs
            mag = F2
            # corner discs only half of vertical chain get half horizontal?
            # if you actually need half-force on top/bottom corner, 
            # you can check row index i similarly.
            fx = +mag if j==0 else -mag
            b.ApplyForce((fx,0), b.worldCenter, wake=True)

    # step world
    w.Step(TIME_STEP, VEL_IT, POS_IT)

    # --- draw ---
    screen.fill((0,0,0))
    # discs
    for b in bodies:
        pygame.draw.circle(screen, (200,200,200),
                          to_screen(b.position),
                          int(DB/2*PPM))
    # bottom platen
    for f in bottom.fixtures:
        vs = [bottom.transform*v for v in f.shape.vertices]
        pts = [to_screen(v) for v in vs]
        pygame.draw.polygon(screen, (80,80,255), pts)
    # top platen
    for f in top.fixtures:
        vs = [top.transform*v for v in f.shape.vertices]
        pts = [to_screen(v) for v in vs]
        pygame.draw.polygon(screen, (255,80,80), pts)

    pygame.display.flip()
    clock.tick(60)

    # ——— Logging ———
    # δ/L
    h_now    = top.position.y - (miny - 0.01)
    delta_L  = (L0 - h_now)/L0

    # σ1 from vertical force on bottom platen
    F1 = 0.0
    for contact in w.contacts:
        # only count contacts touching and involving bottom fixture
        if not contact.touching:
            continue
        a = contact.fixtureA.body
        b = contact.fixtureB.body
        if a is bottom or b is bottom:
            # use normalImpulse ≈ force * timeStep so this is approximate
            # here we extract the normal direction (should be (0,1)) and impulse
            F1 += abs(contact.normal.y * contact.normalImpulse)

    # convert to stress ratio
    sigma_ratio = (F1 / (COLS*DB*1.0)) / SIG2

    with open('log.csv','a', newline='') as f:
        csv.writer(f).writerow((delta_L, sigma_ratio))
