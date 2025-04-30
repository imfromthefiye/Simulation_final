import math
import random
import time
from typing import List, Tuple, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from Box2D import (
    b2World,
    b2FixtureDef,
    b2PolygonShape,
    b2CircleShape,
    b2PrismaticJointDef,
    b2ContactListener,
    b2ContactImpulse,
)
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("burrow_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BurrowEnv")

# Environment constants
FPS = 60
TIME_STEP = 1.0 / FPS
VEL_ITERS, POS_ITERS = 100, 3

CHAMBER_WIDTH = 0.80
CHAMBER_HEIGHT = 1.00
MARGIN = 0.1
SCALE = 400.0
VIEWPORT_W = int((CHAMBER_WIDTH + 2*MARGIN) * SCALE)
VIEWPORT_H = int((CHAMBER_HEIGHT + 2*MARGIN) * SCALE)

D_MIN, D_MAX = 0.01, 0.02
MAX_PARTICLES = 3000
SETTLE_GRAVITY = 0.1
NORMAL_GRAVITY = 9.81

D_PROBE = 0.05
SHAFT_LENGTH = 0.50
TIP_HEIGHT = D_PROBE * 0.8
PRISMATIC_LIMIT = TIP_HEIGHT
EXP_RATIO = 1.2


class ContactDetector(b2ContactListener):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.tip_force = 0.0
        self.shaft_force = 0.0

    def PostSolve(self, contact, impulse: b2ContactImpulse):
        try:
            body_a = contact.fixtureA.body
            body_b = contact.fixtureB.body
            total_impulse = sum(impulse.normalImpulses)
            if self.env.tip and (body_a == self.env.tip or body_b == self.env.tip):
                self.tip_force += total_impulse
            if self.env.shaft and (body_a == self.env.shaft or body_b == self.env.shaft):
                self.shaft_force += total_impulse
        except Exception as e:
            print(f"Error in PostSolve: {e}")

    def EndContact(self, contact):
        self.tip_force = 0.0
        self.shaft_force = 0.0

class BurrowEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, render_mode="none"):
        super().__init__()
        self.render_mode = render_mode
        
        # Initialize pygame differently based on render mode
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            self.clock = pygame.time.Clock()
        else:
            # For headless mode, only initialize the event system
            pygame.init()
            pygame.display.init()
            pygame.display.set_mode((1,1), pygame.NOFRAME)  # Minimal hidden window
            self.screen = None
            self.clock = None

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 1.0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, EXP_RATIO, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.world: Optional[b2World] = None
        self.tip = self.shaft = self.joint = None
        self.chambers = []
        self.grains = []
        self.contact_listener = ContactDetector(self)
        self.grain_height = 0.0
        self.max_penetration = 0.0  # Will be set based on grain height

    def get_grain_height(self) -> float:
        """Calculate current height of grain pile"""
        if not self.grains:
            return 0.0
            
        max_height = 0.0
        settled_count = 0
        
        for grain in self.grains:
            pos_y = grain.position.y
            vel = math.hypot(grain.linearVelocity.x, grain.linearVelocity.y)
            
            if vel < 0.05:  # Consider grain settled
                settled_count += 1
                max_height = max(max_height, pos_y + grain.fixtures[0].shape.radius)
        
        return max_height

    def reset(self, seed=None, options=None):
        self.world = b2World(gravity=(0.0, -NORMAL_GRAVITY))
        self.world.contactListener = self.contact_listener

        self._create_chamber()
        self.grains = rain_grains(self.world)
        
        # Get settled grain height and set max penetration
        self.grain_height = self.get_grain_height()
        self.max_penetration = 0.8 * self.grain_height  # 80% of grain height
        print(f"Grain height: {self.grain_height:.3f}m")
        print(f"Max penetration: {self.max_penetration:.3f}m")

        self.time_elapsed = 0.0
        self.spawned_probe = False
        self.initial_done = False
        self.shaft_ratio = 1.0
        self.prev_shaping = 0.0

        return np.zeros(5, dtype=np.float32), {}

    def _create_chamber(self):
        half_w = CHAMBER_WIDTH / 2
        thickness = 0.01
        self.chambers = []
        self.chambers.append(self.world.CreateStaticBody(
            position=(-half_w + thickness/2, CHAMBER_HEIGHT/2),
            fixtures=b2FixtureDef(shape=b2PolygonShape(box=(thickness/2, CHAMBER_HEIGHT/2)), friction=0.3)))
        self.chambers.append(self.world.CreateStaticBody(
            position=( half_w - thickness/2, CHAMBER_HEIGHT/2),
            fixtures=b2FixtureDef(shape=b2PolygonShape(box=(thickness/2, CHAMBER_HEIGHT/2)), friction=0.3)))
        self.chambers.append(self.world.CreateStaticBody(
            position=(0.0, thickness/2),
            fixtures=b2FixtureDef(shape=b2PolygonShape(box=(half_w, thickness/2)), friction=0.5)))

    def _create_probe(self):
        base_y = CHAMBER_HEIGHT/2
        self.shaft = self.world.CreateDynamicBody(
            position=(0.0, base_y + SHAFT_LENGTH/2),
            fixtures=b2FixtureDef(shape=b2PolygonShape(box=(D_PROBE/2, SHAFT_LENGTH/2)), density=1.0, friction=0.2))
        self.shaft.gravityScale = 0.0

        verts = [(-D_PROBE/2, TIP_HEIGHT/3), (D_PROBE/2, TIP_HEIGHT/3), (0.0, -2*TIP_HEIGHT/3)]
        self.tip = self.world.CreateDynamicBody(
            position=(0.0, base_y - TIP_HEIGHT/3),
            fixtures=b2FixtureDef(shape=b2PolygonShape(vertices=verts), density=1.0, friction=0.2))
        self.tip.gravityScale = 0.0

        self.joint = self.world.CreateJoint(b2PrismaticJointDef(
            bodyA=self.shaft, bodyB=self.tip,
            localAnchorA=(0.0, -SHAFT_LENGTH/2),
            localAnchorB=(0.0, TIP_HEIGHT/3),
            localAxisA=(0.0, -1.0),
            enableLimit=True, lowerTranslation=0.0, upperTranslation=PRISMATIC_LIMIT,
            enableMotor=False, maxMotorForce=1000.0, collideConnected=False))

    def step(self, action):
        self.time_elapsed += TIME_STEP

        if not self.spawned_probe and self.time_elapsed > 4.0:
            self._create_probe()
            self.spawned_probe = True

        if self.spawned_probe and not self.initial_done:
            self.joint.enableMotor = True
            self.joint.motorSpeed = 0.1
            self.world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
            
            # Use adaptive max penetration
            if self.joint.translation >= self.max_penetration:
                self.joint.enableMotor = False
                self.initial_done = True
                print(f"Initial penetration complete at depth: {self.joint.translation:.3f}m")
            
            self._render()
            return np.zeros(5, dtype=np.float32), 0.0, False, False, {}

        if self.spawned_probe and self.initial_done and action is not None:
            if action == 0:
                self._expand_shaft()
                self.shaft_ratio = EXP_RATIO
            elif action == 1:
                self._contract_shaft()
                self.shaft_ratio = 1.0
            elif action == 2:
                self._extend_tip()
            elif action == 3:
                self._move_body_down()

        self.world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)

        depth = (CHAMBER_HEIGHT/2 - self.tip.position.y) / self.max_penetration
        depth = max(0.0, min(depth, 1.0))
        tip_ext = self.joint.translation / PRISMATIC_LIMIT
        F_tip = self.contact_listener.tip_force / 1000.0
        F_shaft = self.contact_listener.shaft_force / 100.0
        state = [depth, self.shaft_ratio, tip_ext, F_tip, F_shaft]

        shaping = depth - 0.1*(self.shaft_ratio - 1)**2 - 0.1*tip_ext**2 - 0.1*F_tip
        reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        terminated = depth >= 1.0

        self._render()
        return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def _expand_shaft(self):
        for f in list(self.shaft.fixtures): self.shaft.DestroyFixture(f)
        self.shaft.CreateFixture(
            shape=b2PolygonShape(box=(D_PROBE*EXP_RATIO/2, SHAFT_LENGTH/2)),
            density=1.0, friction=0.2)

    def _contract_shaft(self):
        for f in list(self.shaft.fixtures): self.shaft.DestroyFixture(f)
        self.shaft.CreateFixture(
            shape=b2PolygonShape(box=(D_PROBE/2, SHAFT_LENGTH/2)),
            density=1.0, friction=0.2)

    def _extend_tip(self, speed=0.2):
        self.joint.enableMotor = True
        self.joint.motorSpeed = speed

    def _move_body_down(self):
        self.joint.enableMotor = False
        d = self.joint.translation
        self.shaft.position = (self.shaft.position.x, self.shaft.position.y - d)
        self.tip.position = (self.tip.position.x, self.tip.position.y - d)

    def _world_to_screen(self, pt):
        x, y = pt
        px = int((x + CHAMBER_WIDTH/2 + MARGIN) * SCALE)
        py = int((CHAMBER_HEIGHT + MARGIN - y) * SCALE)
        return px, py

    def _render(self):
        if self.render_mode != "human":
            return
        self.screen.fill((255,255,255))
        for b in self.chambers:
            for f in b.fixtures:
                pts = [self._world_to_screen((b.transform * v)) for v in f.shape.vertices]
                pygame.draw.polygon(self.screen, (50, 50, 50), pts)
        for g in self.grains:
            px, py = self._world_to_screen((g.position.x, g.position.y))
            r = int(g.fixtures[0].shape.radius * SCALE)
            pygame.draw.circle(self.screen, (150, 150, 150), (px, py), r)
        if self.tip and self.shaft:
            for f in self.shaft.fixtures:
                pts = [self._world_to_screen((self.shaft.transform * v)) for v in f.shape.vertices]
                pygame.draw.polygon(self.screen, (100, 100, 255), pts)
            for f in self.tip.fixtures:
                pts = [self._world_to_screen((self.tip.transform * v)) for v in f.shape.vertices]
                pygame.draw.polygon(self.screen, (255, 100, 100), pts)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
        self.clock.tick(FPS)

    def run(self):
        """Run a demo simulation of the environment."""
        self.reset()
        running = True
        
        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                    
            # Step physics
            self.world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
            
            # Render
            self._render()
            
            # Control frame rate
            self.clock.tick(FPS)

    def close(self):
        if self.world:
            # Cleanup Box2D objects
            if self.shaft:
                self.world.DestroyBody(self.shaft)
            if self.tip:
                self.world.DestroyBody(self.tip)
            for body in self.chambers + self.grains:
                self.world.DestroyBody(body)
        self.world = None
        
        # Cleanup pygame
        pygame.display.quit()
        pygame.quit()

def rain_grains(world: b2World) -> List[object]:
    """Create and settle grains with safe stepping."""
    grains = []
    half_w = CHAMBER_WIDTH/2 - D_MAX/2
    inlet_y = CHAMBER_HEIGHT + 0.05
    world.gravity = (0.0, -SETTLE_GRAVITY)
    
    # Larger batch size for more grains
    BATCH_SIZE = 100  # Increased from 50
    for batch in range(0, MAX_PARTICLES, BATCH_SIZE):
        pygame.event.pump()  # Keep window responsive
        
        # Create batch of grains
        for _ in range(min(BATCH_SIZE, MAX_PARTICLES - batch)):
            r = random.uniform(D_MIN/2, D_MAX/2)
            x = random.uniform(-half_w, half_w)
            g = world.CreateDynamicBody(
                position=(x, inlet_y),
                fixtures=[b2FixtureDef(
                    shape=b2CircleShape(radius=r),
                    density=1.0,
                    friction=0.4
                )],
                linearDamping=0.5
            )
            grains.append(g)
        
        # Fewer steps per batch but more frequent
        for _ in range(5):  # Reduced from 10
            world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
    
    # Increased max settle steps for more grains
    MAX_SETTLE_STEPS = 500  # Increased from 300
    step_count = 0
    while step_count < MAX_SETTLE_STEPS:
        pygame.event.pump()
        world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
        
        # Check if grains have settled
        max_v = max(math.hypot(g.linearVelocity.x, g.linearVelocity.y) for g in grains)
        if max_v < 0.05:  # Velocity threshold for settling
            break
        step_count += 1
    
    world.gravity = (0.0, -NORMAL_GRAVITY)
    for g in grains:
        g.linearDamping = 0.0
    return grains

def heuristic(obs, epsilon=0.2):
    """Improved heuristic with exploration for RL-like behavior
    
    Args:
        obs: Environment observation (depth, shaft_ratio, tip_ext, F_tip, F_shaft)
        epsilon: Exploration rate (probability of taking random action)
    
    Returns:
        action: Selected action (0-3)
    """
    # Random exploration (epsilon-greedy approach)
    if random.random() < epsilon:
        action = random.randint(0, 3)
        logger.debug(f"Taking random exploration action: {action}")
        return action
        
    # Otherwise follow heuristic policy
    depth, shaft_ratio, tip_ext, F_tip, F_shaft = obs
    
    # Log current state
    logger.debug(f"State: depth={depth:.3f}, shaft_ratio={shaft_ratio:.2f}, "
                f"tip_ext={tip_ext:.3f}, F_tip={F_tip:.3f}, F_shaft={F_shaft:.3f}")
    
    # If tip is retracted and forces are low, extend tip
    if tip_ext < 0.1 and F_tip < 0.3:
        action = 2  # extend tip
        logger.debug("Heuristic: extending tip (low resistance)")
    
    # If tip encounters high resistance, expand shaft
    elif F_tip > 0.5 and shaft_ratio < 1.1:
        action = 0  # expand shaft
        logger.debug("Heuristic: expanding shaft (high tip resistance)")
    
    # If shaft is expanded and tip force is moderate, move down
    elif shaft_ratio > 1.1 and F_tip < 0.5:
        action = 3  # move down
        logger.debug("Heuristic: moving down (good conditions)")
    
    # If shaft force is high, contract shaft
    elif F_shaft > 0.5:
        action = 1  # contract shaft
        logger.debug("Heuristic: contracting shaft (high shaft resistance)")
    
    # Default: move down if no other conditions are met
    else:
        action = 3  # move down
        logger.debug("Heuristic: default action - moving down")
    
    return action

def demo_heuristic_burrow(env, seed=None, render=False, episodes=5):
    """Enhanced headless demo with extensive logging
    
    Args:
        env: BurrowEnv instance
        seed: Random seed
        render: Whether to render (should be False for headless)
        episodes: Number of episodes to run
    """
    logger.info(f"Starting headless simulation with seed={seed}, episodes={episodes}")
    
    # Track metrics across episodes
    episode_rewards = []
    episode_steps = []
    episode_depths = []
    
    for episode in range(episodes):
        start_time = time.time()
        logger.info(f"\n=== Episode {episode+1}/{episodes} ===")
        
        # Reset environment
        obs, _ = env.reset(seed=seed)
        logger.info(f"Initial state: {obs}")
        
        # Episode tracking
        total_reward = 0
        steps = 0
        actions_taken = {0: 0, 1: 0, 2: 0, 3: 0}  # Count of each action type
        
        # Main loop
        done = False
        while not done:
            # Get action from heuristic
            action = heuristic(obs)
            actions_taken[action] += 1
            
            # Take step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            # Log progress periodically
            if steps % 10 == 0:
                logger.info(f"Step {steps:4d} - Action: {action} ({['expand','contract','extend','move'][action]})")
                logger.info(f"  Depth: {next_obs[0]:.3f} m")
                logger.info(f"  Shaft ratio: {next_obs[1]:.3f}")
                logger.info(f"  Tip extension: {next_obs[2]:.3f}")
                logger.info(f"  Tip force: {next_obs[3]:.3f}")
                logger.info(f"  Shaft force: {next_obs[4]:.3f}")
                logger.info(f"  Reward: {reward:+.3f}")
            
            # Update for next iteration
            obs = next_obs
            done = terminated or truncated
        
        # Episode summary
        duration = time.time() - start_time
        logger.info(f"\n=== Episode {episode+1} Summary ===")
        logger.info(f"Total steps: {steps}")
        logger.info(f"Final depth: {obs[0]:.3f} m")
        logger.info(f"Total reward: {total_reward:+.3f}")
        logger.info(f"Actions: expand={actions_taken[0]}, contract={actions_taken[1]}, " 
                   f"extend={actions_taken[2]}, move={actions_taken[3]}")
        logger.info(f"Episode duration: {duration:.2f} seconds")
        
        # Store metrics
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_depths.append(obs[0])
    
    # Overall summary
    logger.info("\n=== Simulation Summary ===")
    logger.info(f"Episodes: {episodes}")
    logger.info(f"Avg reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    logger.info(f"Avg steps: {np.mean(episode_steps):.1f} ± {np.std(episode_steps):.1f}")
    logger.info(f"Avg depth: {np.mean(episode_depths):.3f} ± {np.std(episode_depths):.3f}")
    logger.info("Simulation complete!")

if __name__ == "__main__":
    try:
        logger.info("Initializing BurrowEnv in headless mode...")
        env = BurrowEnv(render_mode="none")
        logger.info("Environment created successfully")
        logger.info("Starting headless simulation...")
        
        # Run multiple episodes to gather statistics
        demo_heuristic_burrow(env, render=False, episodes=3)
        
    except Exception as e:
        logger.error(f"Simulation error: {e}", exc_info=True)
    finally:
        env.close()
        logger.info("Environment closed")

