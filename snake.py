import pygame
import numpy as np
from enum import Enum
from collections import namedtuple
from typing import Optional, Tuple

Point = namedtuple('Point', 'x, y')

class Direction(Enum):
    RIGHT = (1, 0)   # (dx, dy)
    LEFT = (-1, 0)
    UP = (0, -1)
    DOWN = (0, 1)

    @property
    def dx(self) -> int:
        return self.value[0] * BLOCK_SIZE

    @property
    def dy(self) -> int:
        return self.value[1] * BLOCK_SIZE

    def turn_right(self) -> 'Direction':
        """Return the direction after turning right."""
        rotations = {
            Direction.RIGHT: Direction.DOWN,
            Direction.DOWN: Direction.LEFT,
            Direction.LEFT: Direction.UP,
            Direction.UP: Direction.RIGHT
        }
        return rotations[self]

    def turn_left(self) -> 'Direction':
        """Return the direction after turning left."""
        rotations = {
            Direction.RIGHT: Direction.UP,
            Direction.UP: Direction.LEFT,
            Direction.LEFT: Direction.DOWN,
            Direction.DOWN: Direction.RIGHT
        }
        return rotations[self]

BLOCK_SIZE = 20
SPEED = 40

WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN1 = (0, 200, 0)
GREEN2 = (0, 150, 0)
BLACK = (0, 0, 0)


class SnakeGame:
    def __init__(self, w: int = 640, h: int = 480, speed: int = SPEED, headless: bool = False) -> None:
        self.w = w
        self.h = h
        self.speed = speed
        self.headless = headless

        if not headless:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()
        else:
            self.display = None
            self.clock = None

        self.reset()

    def reset(self) -> np.ndarray:
        self.direction = Direction.RIGHT

        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

        return self.get_state()

    def cleanup(self) -> None:
        """Clean up pygame resources."""
        if not self.headless:
            pygame.quit()

    def _place_food(self) -> None:
        x = np.random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = np.random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action: int) -> Tuple[float, bool, int]:
        self.frame_iteration += 1

        if not self.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # Store old head position for distance calculation
        old_head = self.head

        self._move(action)
        self.snake.insert(0, self.head)

        # Check game over
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            return -10, True, self.score

        # Check food
        reward = 0
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            
            # TODO: can we improve our reward function? 

        if not self.headless:
            self._update_ui()
            self.clock.tick(self.speed)

        return reward, False, self.score

    def is_collision(self, pt: Optional[Point] = None) -> bool:
        pt = pt or self.head
        return (pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0 or
                pt in self.snake[1:])

    def _update_ui(self) -> None:
        self.display.fill(BLACK)

        for i, pt in enumerate(self.snake):
            color = GREEN1 if i % 2 == 0 else GREEN2
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action: int) -> None:
        # Update direction based on action: 0=straight, 1=right, 2=left
        if action == 1:
            self.direction = self.direction.turn_right()
        elif action == 2:
            self.direction = self.direction.turn_left()

        self.head = Point(self.head.x + self.direction.dx,
                         self.head.y + self.direction.dy)

    def get_state(self) -> np.ndarray:
        # Danger in relative directions (straight, right, left)
        dir_right = self.direction.turn_right()
        dir_left = self.direction.turn_left()

        # Taxicab distance to food
        taxicab_distance = abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)

        return np.array([
            # TODO: implement state variables... 12 of them ideally :0
        ], dtype=float)
