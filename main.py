from __future__ import annotations

import sys

import pygame

from agents import ghost_heuristic, ghost_mcts, ghost_random
from game.difficulty import get_config, performance_score, update_difficulty
from game.maze_generator import PELLET, POWER, WALL, generate_maze
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


def main() -> int:
    pygame.init()
    pygame.display.set_caption("Adaptive Pacman-like Game AI")

    screen_width = GRID_WIDTH * CELL_SIZE
    screen_height = GRID_HEIGHT * CELL_SIZE + 72
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 18)

    difficulty = 2
    env = create_environment(difficulty)
    current_action = "STAY"
    last_performance: float | None = None

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
                elif event.key in KEY_TO_ACTION:
                    current_action = KEY_TO_ACTION[event.key]

        if not env.done:
            legal = env.legal_actions_for_pacman()
            pacman_action = current_action if current_action in legal else "STAY"
            config = get_config(difficulty)
            ghost_actions = choose_ghost_actions(env, config.ghost_agent)
            env.step(pacman_action, ghost_actions)
            for _ in range(config.ghost_speed - 1):
                if env.done:
                    break
                env.step("STAY", choose_ghost_actions(env, config.ghost_agent))
        else:
            last_performance = performance_score(
                env.pellets_collected,
                env.total_pellets,
                env.steps,
                env.max_steps,
                env.won,
            )
            difficulty = update_difficulty(difficulty, last_performance)
            env = create_environment(difficulty)
            current_action = "STAY"

        draw(screen, font, env, difficulty, last_performance)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    return 0


def create_environment(difficulty: int) -> PacmanEnv:
    config = get_config(difficulty)
    generated = generate_maze(GRID_WIDTH, GRID_HEIGHT, config)
    return PacmanEnv(
        generated.grid,
        generated.pacman_start,
        generated.ghost_starts,
        max_steps=GRID_WIDTH * GRID_HEIGHT * 2,
    )


def choose_ghost_actions(env: PacmanEnv, agent_name: str) -> list[str]:
    actions = []
    for ghost_id in range(len(env.ghost_positions)):
        if agent_name == "mcts":
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
        f"({config.ghost_agent}) | Pellets: {env.pellets_collected}/{env.total_pellets}"
    )
    screen.blit(font.render(score_text, True, TEXT), (12, GRID_HEIGHT * CELL_SIZE + 10))

    perf = "n/a" if last_performance is None else f"{last_performance:.2f}"
    help_text = f"Move: arrows/WASD | R: new maze | Last performance: {perf}"
    screen.blit(font.render(help_text, True, TEXT), (12, GRID_HEIGHT * CELL_SIZE + 38))


if __name__ == "__main__":
    raise SystemExit(main())
