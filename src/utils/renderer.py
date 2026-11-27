import math
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jax import random

from env.base import RADIUS


CANVAS_SIZE = (1080, 650)
BACKGROUND = (20, 28, 38)
PANEL = (29, 41, 52)
BORDER = (57, 74, 87)
FIELD = (24, 36, 47)
GRID = (34, 50, 62)
GRID_MAJOR = (44, 62, 75)
TEXT = (231, 239, 243)
MUTED = (164, 185, 198)
ROUTE = (108, 169, 206)
AGENT = (152, 216, 224)
GOAL = (144, 202, 172)
STATIC = (113, 136, 154)
MOVING = (226, 175, 111)
SENSOR = (131, 186, 202)


class PygameFrontend:
    def __init__(self, env, env_params, init_state, eval_mode=False, agent_fn=None,
                 video_path=None, record_video=True):
        pygame.display.init()
        pygame.font.init()
        self.env = env
        self.params = env_params
        self.init_state = init_state
        # AutoResetWrapper stores the actual environment class in ``env``.
        base_env = getattr(env, "env", env)
        self.vehicle_controls = bool(getattr(base_env, "uses_vehicle_controls", False))
        self.eval_mode = eval_mode
        self.agent_fn = agent_fn
        self.video_writer = None
        self.video_path = Path(video_path) if video_path else None
        if record_video:
            if self.video_path is None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.video_path = Path("artifacts") / f"simulator_{stamp}" / "episode.mp4"
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            self.video_writer = imageio.get_writer(
                str(self.video_path), fps=int(env_params.fps), codec="libx264",
                macro_block_size=1,
            )
            print(f"Recording simulator video: {self.video_path}")
        self.screen = pygame.display.set_mode(CANVAS_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption("IAEML | Driving simulator")
        self.canvas = pygame.Surface(CANVAS_SIZE).convert()
        self.sensor_layer = pygame.Surface(CANVAS_SIZE, pygame.SRCALPHA)
        self.fonts = {
            "title": pygame.font.Font(None, 41),
            "heading": pygame.font.Font(None, 26),
            "label": pygame.font.Font(None, 21),
            "small": pygame.font.Font(None, 18),
        }

        self.map_panel = pygame.Rect(24, 92, 728, 499)
        self.telemetry_panel = pygame.Rect(770, 92, 286, 499)
        self.map_height, self.map_width = map(int, env_params.map_height_width)
        map_space = pygame.Rect(43, 151, 690, 365)
        self.cell_size = min(
            map_space.width / self.map_width, map_space.height / self.map_height
        )
        self.field = pygame.Rect(
            0, 0, round(self.map_width * self.cell_size), round(self.map_height * self.cell_size)
        )
        self.field.center = map_space.center
        self.sensor_layer.set_clip(self.field)
        self.grid = self.make_grid()

        self.clock = pygame.time.Clock()
        self.key = random.PRNGKey(0)
        self.running = True
        self.paused = False
        self.show_rays = True
        self.reset()

    def reset(self):
        self.obs, self.state = self.env.reset(self.key, self.params, self.init_state)
        self.reward = 0.0
        self.info = {}

    def world_to_screen(self, position):
        world_x, world_y = position
        return (
            round(self.field.left + (world_x + 0.5) * self.cell_size),
            round(self.field.top + (self.map_height - 0.5 - world_y) * self.cell_size),
        )

    def make_grid(self):
        grid = pygame.Surface(self.field.size).convert()
        grid.fill(FIELD)
        for column in range(1, self.map_width):
            x = round(column * self.cell_size)
            color = GRID_MAJOR if column % 5 == 0 else GRID
            pygame.draw.line(grid, color, (x, 0), (x, grid.get_height()))
        for row in range(1, self.map_height):
            y = round(row * self.cell_size)
            color = GRID_MAJOR if row % 5 == 0 else GRID
            pygame.draw.line(grid, color, (0, y), (grid.get_width(), y))
        return grid

    def text(self, value, position, style="label", color=TEXT, align_right=False):
        label = self.fonts[style].render(value, True, color)
        if align_right:
            rect = label.get_rect(topright=position)
        else:
            rect = label.get_rect(topleft=position)
        self.canvas.blit(label, rect)

    def draw_panel(self, rect):
        pygame.draw.rect(self.canvas, PANEL, rect, border_radius=12)
        pygame.draw.rect(self.canvas, BORDER, rect, width=1, border_radius=12)

    def draw_header(self):
        self.text("IAEML  /  DRIVING LAB", (24, 18), "small", MUTED)
        self.text("Driving simulator", (24, 40), "title")
        badge = pygame.Rect(889, 30, 167, 34)
        self.draw_panel(badge)
        color = MOVING if self.paused else GOAL
        pygame.draw.circle(self.canvas, color, (906, 47), 4)
        if self.paused:
            status = "Paused"
        elif self.eval_mode and self.agent_fn is not None:
            status = "Policy control"
        else:
            status = "Manual control"
        self.text(status, (920, 40), "small")

    def draw_map(self, state, rays, ray_distances):
        self.draw_panel(self.map_panel)
        self.text("Navigation map", (43, 111), "heading")
        self.text(
            f"{self.map_width} × {self.map_height}",
            (732, 114),
            "small",
            MUTED,
            align_right=True,
        )
        self.canvas.blit(self.grid, self.field)
        self.canvas.set_clip(self.field)
        center = self.world_to_screen(state.agent_pos)
        route = [center, *(self.world_to_screen(point) for point in state.path_array)]
        if len(route) > 1:
            pygame.draw.lines(self.canvas, (54, 86, 109), False, route, 7)
            pygame.draw.lines(self.canvas, ROUTE, False, route, 2)

        tile = max(3, round(self.cell_size * 0.82))
        for position in state.static_obstacles:
            rect = pygame.Rect(0, 0, tile, tile)
            rect.center = self.world_to_screen(position)
            pygame.draw.rect(self.canvas, (49, 65, 78), rect, border_radius=3)
            pygame.draw.rect(self.canvas, (84, 106, 122), rect, width=1, border_radius=3)

        for position in state.kinematic_obstacles:
            self.draw_moving_obstacle(self.world_to_screen(position))

        goal = self.world_to_screen(state.goal_pos)
        goal_radius = max(2, round(self.cell_size * RADIUS))
        pygame.draw.circle(self.canvas, (59, 96, 84), goal, goal_radius, width=1)
        pygame.draw.circle(self.canvas, GOAL, goal, 8, width=2)
        pygame.draw.circle(self.canvas, GOAL, goal, 3)
        if self.show_rays:
            self.draw_rays(state.agent_pos, center, rays, ray_distances)
        self.draw_agent(center, state.agent_forward_dir)
        self.canvas.set_clip(None)
        pygame.draw.rect(self.canvas, BORDER, self.field, width=1)

        legend = (
            ("Agent", AGENT), ("Goal", GOAL), ("Route", ROUTE),
            ("Static", STATIC), ("Moving", MOVING), ("Sensor rays", SENSOR),
        )
        for index, (label, color) in enumerate(legend):
            position = 47 + index * 113
            pygame.draw.circle(self.canvas, color, (position + 4, 556), 4)
            self.text(label, (position + 15, 549), "small", MUTED)

    def draw_moving_obstacle(self, center):
        radius = max(5, round(self.cell_size * 0.32))
        pygame.draw.circle(self.canvas, (103, 77, 53), center, round(self.cell_size * RADIUS), width=1)
        x, y = center
        points = ((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y))
        pygame.draw.polygon(self.canvas, (119, 80, 49), points)
        pygame.draw.polygon(self.canvas, MOVING, points, width=2)

    def draw_rays(self, agent_pos, center, rays, distances):
        self.sensor_layer.fill((0, 0, 0, 0))
        max_distance = float(self.params.perception_radius)
        for ray, distance in zip(rays, distances):
            endpoint = self.world_to_screen(agent_pos + ray)
            hit = distance < max_distance - 0.001
            color = MOVING if hit else SENSOR
            pygame.draw.line(self.sensor_layer, (*color, 118 if hit else 48), center, endpoint)
            if hit:
                pygame.draw.circle(self.sensor_layer, (*color, 190), endpoint, 2)
        self.canvas.blit(self.sensor_layer, (0, 0))

    def draw_agent(self, center, direction):
        heading = np.array([direction[0], -direction[1]])
        magnitude = np.linalg.norm(heading)
        heading = heading / magnitude if magnitude > 0 else np.array([0.0, -1.0])
        side = np.array([-heading[1], heading[0]])
        radius = max(6, min(17, round(self.cell_size * 0.42)))
        origin = np.asarray(center)
        points = [
            tuple(origin + radius * (forward * heading + across * side))
            for forward, across in ((1.0, 0.0), (-0.62, 0.68), (-0.41, 0.0), (-0.62, -0.68))
        ]
        pygame.draw.circle(self.canvas, (54, 87, 104), center, radius + 4, width=1)
        pygame.draw.polygon(self.canvas, AGENT, points)
        pygame.draw.polygon(self.canvas, (225, 249, 250), points, width=1)

    def draw_telemetry(self, state, observation, ray_distances, reward):
        self.draw_panel(self.telemetry_panel)
        left, right = 790, 1036
        self.text("Telemetry", (left, 111), "heading")
        step = int(state.time)
        distance = np.linalg.norm(state.goal_pos - state.agent_pos)
        path_gap = getattr(observation, "distance_to_path", None)
        step_limit = (
            "∞" if self.params.max_steps_in_episode <= 0
            else f"{self.params.max_steps_in_episode:,}"
        )
        metrics = [
            ("Step", f"{step:,} / {step_limit}"),
            ("Sim time", f"{step * self.params.step_size:.2f} s"),
        ]
        if self.vehicle_controls:
            metrics.extend([
                ("Speed", f"{float(state.speed):+.2f} m/s"),
                ("Steering", f"{math.degrees(float(state.steering_angle)):+.1f}°"),
            ])
        metrics.extend([
            ("Goal distance", f"{distance:.2f} m"),
            ("Path offset", f"{float(path_gap):.2f} m" if path_gap is not None else "—"),
            ("Last reward", f"{float(reward):+.2f}"),
        ])
        for index, (label, value) in enumerate(metrics):
            y = 151 + index * 32
            self.text(label, (left, y), "label", MUTED)
            self.text(value, (right, y), "label", align_right=True)
            pygame.draw.line(self.canvas, BORDER, (left, y + 26), (right, y + 26))

        sensors_y = 151 + len(metrics) * 32 + 15
        self.text("Range sensors", (left, sensors_y), "heading")
        status = "Map rays on" if self.show_rays else "Map rays off"
        self.text(f"{len(ray_distances)} sensors   /   {status}", (left, sensors_y + 27), "small", MUTED)
        chart_y = sensors_y + 56
        chart = pygame.Rect(left, chart_y, right - left, 55)
        pygame.draw.rect(self.canvas, FIELD, chart, border_radius=5)
        scale = max(float(self.params.perception_radius), 0.001)
        for index, distance in enumerate(ray_distances):
            start = round(chart.left + index * chart.width / len(ray_distances))
            end = round(chart.left + (index + 1) * chart.width / len(ray_distances))
            if end <= start:
                continue
            height = max(1, round(np.clip(distance / scale, 0, 1) * (chart.height - 10)))
            color = MOVING if distance < scale - 0.001 else SENSOR
            bar = pygame.Rect(start + 1, chart.bottom - 5 - height, max(1, end - start - 2), height)
            pygame.draw.rect(self.canvas, color, bar, border_radius=1)
        self.text("Amber bars mark detected obstacles", (left, chart_y + 61), "small", MUTED)

    def draw(self):
        state, observation, reward = jax.device_get((self.state, self.obs, self.reward))
        rays = np.asarray(observation.collision_rays)
        ray_distances = np.linalg.norm(rays, axis=1)
        self.canvas.fill(BACKGROUND)
        self.draw_header()
        self.draw_map(state, rays, ray_distances)
        self.draw_telemetry(state, observation, ray_distances, reward)
        self.text(
            "W/S Accel/Reverse   A/D Steer   Space Pause   R Reset   V Rays   Esc Exit",
            (24, 616),
            "label",
            MUTED,
        )
        self.present()

    def present(self):
        size = self.screen.get_size()
        if size == CANVAS_SIZE:
            self.screen.blit(self.canvas, (0, 0))
        else:
            target = self.canvas.get_rect().fit(self.screen.get_rect())
            target.width = max(1, target.width)
            target.height = max(1, target.height)
            target.center = self.screen.get_rect().center
            self.screen.fill(BACKGROUND)
            self.screen.blit(pygame.transform.smoothscale(self.canvas, target.size), target)
        pygame.display.flip()
        if self.video_writer is not None:
            # Record the full-resolution simulator canvas, not the resized window.
            frame = pygame.surfarray.array3d(self.canvas).swapaxes(0, 1)
            self.video_writer.append_data(frame)

    def handle_keys(self):
        keys = pygame.key.get_pressed()
        forward = int(keys[pygame.K_UP] or keys[pygame.K_w])
        backward = int(keys[pygame.K_DOWN] or keys[pygame.K_s])
        left = int(keys[pygame.K_LEFT] or keys[pygame.K_a])
        right = int(keys[pygame.K_RIGHT] or keys[pygame.K_d])

        if self.vehicle_controls:
            # The robotaxi action is [acceleration in m/s^2, steering in rad].
            if forward == backward:
                acceleration = 0.0
            elif forward:
                acceleration = float(self.params.max_acceleration)
            else:
                acceleration = -float(self.params.max_reverse_acceleration)
            steering = (
                left * float(self.params.max_steering_angle)
                - right * float(self.params.max_steering_angle)
            )
            return jnp.array([acceleration, steering], dtype=jnp.float32)

        # Keep the original direction-control interface for BaseEnv demos.
        x = right - left
        y = forward - backward
        scale = math.hypot(x, y) or 1
        return jnp.array([x / scale, y / scale], dtype=jnp.float32)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_r:
                self.reset()
            elif event.key == pygame.K_v:
                self.show_rays = not self.show_rays

    def run(self):
        try:
            while self.running:
                for event in pygame.event.get():
                    self.handle_event(event)
                    if not self.running:
                        break
                if not self.running:
                    break
                if not self.paused:
                    if self.eval_mode and self.agent_fn is not None:
                        action = self.agent_fn(self.state)
                    else:
                        action = self.handle_keys()
                    self.obs, self.state, self.reward, _, self.info = self.env.step(
                        self.key, self.state, action, self.params
                    )
                self.draw()
                self.clock.tick(self.params.fps)
        finally:
            if self.video_writer is not None:
                self.video_writer.close()
                print(f"Video saved: {self.video_path}")
            pygame.quit()
