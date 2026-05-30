from __future__ import annotations

import random
import sys

import pygame

from agents import ghost_heuristic, ghost_mcts, ghost_random
from game.difficulty import PlayerProfile, RunMetrics, get_config, performance_score
from game.maze_generator import PELLET, POWER, WALL, generate_balanced_maze
from game.pacman_env import PacmanEnv

CELL_SIZE = 28
GRID_WIDTH = 21
GRID_HEIGHT = 21
FPS = 10

BLACK = (8, 10, 18)
WALL_BLUE = (42, 91, 214)
PATH = (16, 19, 31)
PELLET_COLOR = (246, 217, 139)
POWER_COLOR = (126, 224, 170)
PACMAN_YELLOW = (255, 213, 64)
GHOST_RED = (234, 74, 91)
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

    difficulty = 2
    profile = PlayerProfile(history_size=5)
    env = create_environment(difficulty)
    current_action = "STAY"
    desired_action: str | None = None
    last_performance: float | None = None
    last_win_rate: float | None = None
    last_pellet_completion: float | None = None

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
                    current_action = "STAY"
                    desired_action = None
                elif event.key in KEY_TO_ACTION:
                    desired_action = KEY_TO_ACTION[event.key]
            elif event.type == pygame.KEYUP and event.key in KEY_TO_ACTION:
                released_action = KEY_TO_ACTION[event.key]
                pressed = pygame.key.get_pressed()
                if desired_action == released_action and not action_is_pressed(pressed, released_action):
                    desired_action = None

        if not env.done:
            legal = env.legal_actions_for_pacman()
            pressed = pygame.key.get_pressed()
            if desired_action is not None and not action_is_pressed(pressed, desired_action):
                desired_action = None

            if desired_action in legal:
                pacman_action = desired_action
            elif current_action in legal and action_is_pressed(pressed, current_action):
                pacman_action = current_action
            else:
                pacman_action = "STAY"
            current_action = pacman_action

            config = get_config(difficulty)
            ghost_actions = choose_ghost_actions(env, config.ghost_agent, config.ghost_aggression)
            env.step(pacman_action, ghost_actions)
            for _ in range(config.ghost_speed - 1):
                if env.done:
                    break
                env.step("STAY", choose_ghost_actions(env, config.ghost_agent, config.ghost_aggression))
        else:
            config = get_config(difficulty)
            last_performance = performance_score(
                env.pellets_collected,
                env.total_pellets,
                env.steps,
                env.max_steps,
                env.won,
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
            current_action = "STAY"
            desired_action = None

        draw(
            screen,
            font,
            env,
            difficulty,
            last_performance,
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
    return PacmanEnv(
        generated.grid,
        generated.pacman_start,
        generated.ghost_starts,
        max_steps=GRID_WIDTH * GRID_HEIGHT * 2,
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
            actions.append(ghost_heuristic.choose_action(env, ghost_id))
        else:
            actions.append(ghost_random.choose_action(env, ghost_id))
    return actions


def draw(
    screen: pygame.Surface,
    font: pygame.font.Font,
    env: PacmanEnv,
    difficulty: int,
    last_performance: float | None,
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

    px, py = env.pacman_pos
    pygame.draw.circle(
        screen,
        PACMAN_YELLOW,
        (px * CELL_SIZE + CELL_SIZE // 2, py * CELL_SIZE + CELL_SIZE // 2),
        CELL_SIZE // 2 - 4,
    )

    for gx, gy in env.ghost_positions:
        ghost_rect = pygame.Rect(gx * CELL_SIZE + 5, gy * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10)
        pygame.draw.rect(screen, GHOST_RED, ghost_rect, border_radius=5)

    config = get_config(difficulty)
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
