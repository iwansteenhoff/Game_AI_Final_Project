from __future__ import annotations

import random
import sys

import sys
import pygame
import math
from agents import ghost_heuristic, ghost_mcts, ghost_random

import game
from game.difficulty import PlayerProfile, RunMetrics, get_config, performance_score
from game.maze_generator import PELLET, POWER, WALL, generate_balanced_maze, solve
from game.pacman_env import PacmanEnv


CELL_SIZE = 28
GRID_WIDTH = 21
GRID_HEIGHT = 21
FPS = 60  # Higher FPS for smooth movement

# Movement speeds (pixels per frame at 60 FPS)
PACMAN_SPEED = 0.1
GHOST_SPEED = 0.1
GHOST_FRIGHTENED_SPEED = 2.0

BLACK = (8, 10, 18)
WALL_BLUE = (42, 91, 214)
PATH = (16, 19, 31)
PELLET_COLOR = (246, 217, 139)
POWER_COLOR = (126, 224, 170)
PACMAN_YELLOW = (255, 213, 64)
GHOST_FRIGHTENED = (58, 110, 220)
GHOST_FLASH = (230, 230, 255)
GHOST_RESPAWNING = (122, 128, 145)
RESPAWN_OUTLINE = (255, 105, 180)
SPIKE_COLOR = (255, 100, 40)
TEXT = (238, 242, 255)
PACMAN_YELLOW = (255, 213, 64)
GHOST_RED = (234, 74, 91)
# Add these two new colors:
GHOST_GREEN = (76, 217, 100)
GHOST_ORANGE = (255, 149, 0)
TEXT = (238, 242, 255)

KEY_TO_ACTION = {
    pygame.K_UP: "UP",
    pygame.K_w: "UP",
    pygame.K_DOWN: "DOWN",
    pygame.K_s: "DOWN",
    pygame.K_LEFT: "LEFT",
    pygame.K_a: "LEFT",
    pygame.K_RIGHT: "RIGHT",
    pygame.K_d: "RIGHT",
}

ACTION_KEYS = {
    action: tuple(key for key, mapped_action in KEY_TO_ACTION.items() if mapped_action == action)
    for action in set(KEY_TO_ACTION.values())
}


def main() -> int:
    pygame.init()
    pygame.display.set_caption("Adaptive Pacman-like Game AI")

    screen_width = GRID_WIDTH * CELL_SIZE
    screen_height = GRID_HEIGHT * CELL_SIZE + 96
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 18)

    difficulty = 1
    profile = PlayerProfile(history_size=5)
    env = create_environment(difficulty)

    #  Smooth movement state 
    pacman_pixel_pos = [env.pacman_pos[0] * CELL_SIZE + CELL_SIZE // 2, env.pacman_pos[1] * CELL_SIZE + CELL_SIZE // 2]
    pacman_next_dir = "STAY"  # Buffered input
    ghost_pixel_pos = [
        [gx * CELL_SIZE + CELL_SIZE // 2, gy * CELL_SIZE + CELL_SIZE // 2]
        for gx, gy in env.ghost_positions
    ]

    current_action = "STAY"
    desired_action: str | None = None
    last_performance: float | None = None
    last_win_rate: float | None = None
    last_pellet_completion: float | None = None
    win_streak = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env = create_environment(difficulty)
                elif event.key in KEY_TO_ACTION:
                    # Buffered input: store next intended direction
                    pacman_next_dir = KEY_TO_ACTION[event.key]

        if not env.done:
            config = get_config(difficulty)
            ghost_actions = choose_ghost_actions(env, config.ghost_agent, config.ghost_aggression)
            env.step(pacman_next_dir, ghost_actions)

            pacman_pixel_pos = env.pacman_pixel_pos[:]
            # Update ghost pixel positions
            ghost_pixel_pos = [pos[:] for pos in env.ghost_pixel_pos]

        else:
            config = get_config(difficulty)
            if env.won:
                win_streak += 1
            last_performance = performance_score(
                env.pellets_collected,
                env.total_pellets,
                env.steps,
                env.max_steps,
                env.traveled_steps,
                env.generated_greedy_solution,
                env.won,
                win_streak,
                env.power_pellets_collected,
                env.ghosts_eaten,
            )
            profile.add_run(
                RunMetrics(
                    difficulty,
                    env.won,
                    env.steps,
                    env.max_steps,
                    env.pellets_collected,
                    env.total_pellets,
                    env.power_pellets_collected,
                    env.ghosts_eaten,
                    env.deaths_to_ghost,
                    env.maze_difficulty_score,
                    config.ghost_agent,
                    len(env.ghost_positions),
                    last_performance,
                )
            )
            difficulty = profile.recommended_difficulty(difficulty)
            last_win_rate = profile.recent_win_rate
            last_pellet_completion = profile.recent_pellet_completion
            env = create_environment(difficulty)
            pacman_next_dir = "STAY"
            pacman_pixel_pos = env.pacman_pixel_pos[:]
            ghost_pixel_pos = [pos[:] for pos in env.ghost_pixel_pos]

        #  Drawing 
        draw_smooth(
            screen,
            font,
            env,
            difficulty,
            last_performance,
            pacman_pixel_pos, 
            ghost_pixel_pos,
            profile.recent_performance,
            last_win_rate,
            last_pellet_completion,
        )
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    return 0

def create_environment(difficulty: int) -> PacmanEnv:
    config = get_config(difficulty)
    generated = generate_balanced_maze(GRID_WIDTH, GRID_HEIGHT, config, difficulty)
    generated_greedy_solution = solve(generated)
    print(f"Greedy solution: {generated_greedy_solution}")
    return PacmanEnv(
        generated.grid,
        generated.pacman_start,
        generated.ghost_starts,
      
        generated.ghost_respawn_location,
        max_steps=GRID_WIDTH * GRID_HEIGHT * 20,
        frighten_duration=config.frighten_duration,
        move_speed=PACMAN_SPEED,
      
        generated_greedy_solution=generated_greedy_solution,
        maze_difficulty_score=generated.difficulty_score,
    )


def action_is_pressed(pressed: pygame.key.ScancodeWrapper, action: str) -> bool:
    return any(pressed[key] for key in ACTION_KEYS.get(action, ()))


def choose_ghost_actions(env: PacmanEnv, agent_name: str, aggression: float) -> list[str]:
    actions = []
    for ghost_id in range(len(env.ghost_positions)):
        if random.random() > aggression:
            actions.append(ghost_random.choose_action(env, ghost_id))
        elif agent_name == "mcts":
            actions.append(ghost_mcts.choose_action(env, ghost_id))
        elif agent_name == "heuristic":
            # Pass the difficulty into the heuristic ghost!
            actions.append(ghost_heuristic.choose_action(env, ghost_id))
        else:
            actions.append(ghost_random.choose_action(env, ghost_id))
    return actions


def draw_pacman_at(px, py, screen, env, id=0):
    # Animation (open/close)
    mouth_phase = (pygame.time.get_ticks() // 100) % 4
    mouth_angle = 30 if mouth_phase < 2 else 5

    # Direction → base angle
    dir_to_angle = {
        "RIGHT": 0,
        "DOWN": 270,
        "LEFT": 180,
        "UP": 90,
        "STAY": 0,  # fallback
    }

    base_angle = dir_to_angle.get(env.pacman_dir, 0)

    pacman_rect = pygame.Rect(0, 0, CELL_SIZE - 8, CELL_SIZE - 8)
    pacman_rect.center = (round(px), round(py))
    # --- Draw full yellow body ---
    pygame.draw.circle(
        screen,
        PACMAN_YELLOW,
        pacman_rect.center,
        pacman_rect.width // 2
    )

    # --- Cut out mouth using a triangle ---
    cx, cy = pacman_rect.center
    r = pacman_rect.width - (pacman_rect.width // 2) # rounding up

    angle1 = math.radians(base_angle + mouth_angle)
    angle2 = math.radians(base_angle - mouth_angle)

    p1 = (cx, cy)
    p2 = (cx + 1.1 * r * math.cos(angle1), cy - 1.1 * r * math.sin(angle1))
    p3 = (cx + 1.1 * r * math.cos(angle2), cy - 1.1 * r * math.sin(angle2))

    pygame.draw.polygon(screen, PATH, [p1, p2, p3])

def draw_ghost_at(gx, gy, screen, env, id=0):
    respawn_timer = env.ghost_respawn_timers[id] if id < len(env.ghost_respawn_timers) else 0
    frightened_timer = env.ghost_frighten_timers[id] if id < len(env.ghost_frighten_timers) else 0

    if respawn_timer > 0:
        color = GHOST_RESPAWNING
    elif frightened_timer > 0:
        flash = ((frightened_timer // 15) % 2) == 0
        color = GHOST_FLASH if flash else GHOST_FRIGHTENED
    else:
        color = GHOST_RED
    ghost_rect = pygame.Rect(0, 0, CELL_SIZE - 10, CELL_SIZE - 10)
    ghost_rect.center = (int(gx), int(gy))
    pygame.draw.rect(screen, color, ghost_rect, border_radius=5)
    # Eyes (simple)
    eye_offset = 4
    pygame.draw.circle(screen, (255, 255, 255), (ghost_rect.centerx - eye_offset, ghost_rect.centery - 2), 3)
    pygame.draw.circle(screen, (255, 255, 255), (ghost_rect.centerx + eye_offset, ghost_rect.centery - 2), 3)
    pygame.draw.circle(screen, (0, 0, 255), (ghost_rect.centerx - eye_offset, ghost_rect.centery - 2), 1)
    pygame.draw.circle(screen, (0, 0, 255), (ghost_rect.centerx + eye_offset, ghost_rect.centery - 2), 1)
    
    
#  New draw function for smooth movement and animation 
def draw_smooth(
    screen: pygame.Surface,
    font: pygame.font.Font,
    env: PacmanEnv,
    difficulty: int,
    last_performance: float | None,
    pacman_pixel_pos: list[float],
    ghost_pixel_pos: list[list[float]],
    recent_performance: float | None,
    recent_win_rate: float | None,
    recent_pellet_completion: float | None,
) -> None:
    screen.fill(BLACK)

    for y, row in enumerate(env.grid):
        for x, tile in enumerate(row):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, WALL_BLUE if tile == WALL else PATH, rect)

            center = rect.center
            if tile == PELLET:
                pygame.draw.circle(screen, PELLET_COLOR, center, 3)
            elif tile == POWER:
                pygame.draw.circle(screen, POWER_COLOR, center, 7)

    respawn_rect = pygame.Rect(
        env.ghost_respawn_location[0] * CELL_SIZE,
        env.ghost_respawn_location[1] * CELL_SIZE,
        CELL_SIZE,
        CELL_SIZE,
    )
    pygame.draw.rect(screen, (0, 0, 0), respawn_rect)
    pygame.draw.rect(screen, RESPAWN_OUTLINE, respawn_rect, width=3)

    # Pacman animation (simple mouth open/close) 
    px, py = pacman_pixel_pos
    
    draw_pacman_at(px, py, screen, env)
    

    #  Ghosts (simple animation: eyes) 
    for i, (gx, gy) in enumerate(ghost_pixel_pos):
        draw_ghost_at(gx, gy, screen, env, id=i)

    score_text = (
        f"Difficulty: {difficulty} | Ghosts: {len(env.ghost_positions)} "
        f"({config.ghost_agent} {config.ghost_aggression:.0%}) | Maze: {env.maze_difficulty_score:.2f}"
    )
    screen.blit(font.render(score_text, True, TEXT), (12, GRID_HEIGHT * CELL_SIZE + 10))

    perf = "n/a" if last_performance is None else f"{last_performance:.2f}"
    recent = "n/a" if recent_performance is None else f"{recent_performance:.2f}"
    win_rate = "n/a" if recent_win_rate is None else f"{recent_win_rate:.0%}"
    pellet_rate = "n/a" if recent_pellet_completion is None else f"{recent_pellet_completion:.0%}"
    run_text = (
        f"Pellets: {env.pellets_collected}/{env.total_pellets} | "
        f"Power: {env.power_timer} | Ghosts eaten: {env.ghosts_eaten}"
    )
    profile_text = f"Last: {perf} | Recent: {recent} | Win: {win_rate} | Pellets: {pellet_rate}"
    screen.blit(font.render(run_text, True, TEXT), (12, GRID_HEIGHT * CELL_SIZE + 38))
    screen.blit(font.render(profile_text, True, TEXT), (12, GRID_HEIGHT * CELL_SIZE + 66))

if __name__ == "__main__":
    raise SystemExit(main())
