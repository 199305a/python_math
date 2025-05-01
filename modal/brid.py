import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Game constants
GRAVITY = 0.5
JUMP_FORCE = -8
PIPE_SPEED = -3
PIPE_GAP = 150
PIPE_WIDTH = 50
LAND_HEIGHT = 50
BIRD_SIZE = 20


# Colors
def get_random_light_color():
    return (
        random.randint(200, 255),
        random.randint(200, 255),
        random.randint(200, 255),
    )


def get_random_dark_color():
    return (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))


def get_land_color():
    return random.choice([(139, 69, 19), (255, 200, 0)])


def get_pipe_color():
    return random.choice([(0, 100, 0), (188, 143, 87), (64, 64, 64)])


# Bird class
class Bird:
    def __init__(self):
        self.x = SCREEN_WIDTH // 4
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.shape_type = random.choice(["square", "circle", "triangle"])
        self.color = get_random_dark_color()
        self.size = BIRD_SIZE

    def draw(self, screen):
        half = self.size // 2
        if self.shape_type == "square":
            pygame.draw.rect(
                screen, self.color, (self.x - half, self.y - half, self.size, self.size)
            )
        elif self.shape_type == "circle":
            pygame.draw.circle(screen, self.color, (self.x, self.y), half)
        elif self.shape_type == "triangle":
            points = [
                (self.x, self.y - half),
                (self.x - half, self.y + half),
                (self.x + half, self.y + half),
            ]
            pygame.draw.polygon(screen, self.color, points)

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity


# Pipe class
class Pipe:
    def __init__(self):
        self.x = SCREEN_WIDTH
        self.gap_center = random.randint(100, SCREEN_HEIGHT - 100)
        self.color = get_pipe_color()
        self.passed = False

    def draw(self, screen):
        top_height = self.gap_center - PIPE_GAP // 2
        bottom_y = self.gap_center + PIPE_GAP // 2
        pygame.draw.rect(screen, self.color, (self.x, 0, PIPE_WIDTH, top_height))
        pygame.draw.rect(
            screen, self.color, (self.x, bottom_y, PIPE_WIDTH, SCREEN_HEIGHT - bottom_y)
        )

    def update(self):
        self.x += PIPE_SPEED

    def off_screen(self):
        return self.x + PIPE_WIDTH < 0


# Game variables
bird = Bird()
pipes = []
clock = pygame.time.Clock()
score = 0
best_score = 0
game_over = False
background_color = get_random_light_color()
land_color = get_land_color()
font = pygame.font.Font(None, 36)


def reset_game():
    global bird, pipes, score, game_over, background_color, land_color
    bird = Bird()
    pipes = []
    score = DEFAULT_VELOCITY
    game_over = False
    background_color = get_random_light_color()
    land_color = get_land_color()


# Main loop
running = True
while running:
    screen.fill(background_color)

    # Draw land
    pygame.draw.rect(
        screen, land_color, (0, SCREEN_HEIGHT - LAND_HEIGHT, SCREEN_WIDTH, LAND_HEIGHT)
    )

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                running = False
            if event.key == pygame.K_SPACE:
                if game_over:
                    reset_game()
                else:
                    bird.velocity += JUMP_FORCE

    if not game_over:
        bird.update()
        # Check for collisions
        bird_rect = pygame.Rect(
            bird.x - BIRD_SIZE // 2, bird.y - BIRD_SIZE // 2, BIRD_SIZE, BIRD_SIZE
        )
        if bird.y - BIRD_SIZE // 2 < 0 or bird_rect.colliderect(
            pygame.Rect(0, SCREEN_HEIGHT - LAND_HEIGHT, SCREEN_WIDTH, LAND_HEIGHT)
        ):
            game_over = True

        # Generate pipes
        if len(pipes) == 0 or pipes[-1].x < SCREEN_WIDTH - 300:
            pipes.append(Pipe())

        # Update and draw pipes
        for pipe in pipes[:]:
            pipe.update()
            pipe.draw(screen)
            # Check collision
            top_pipe = pygame.Rect(
                pipe.x, 0, PIPE_WIDTH, pipe.gap_center - PIPE_GAP // 2
            )
            bottom_pipe = pygame.Rect(
                pipe.x,
                pipe.gap_center + PIPE_GAP // 2,
                PIPE_WIDTH,
                SCREEN_HEIGHT - (pipe.gap_center + PIPE_GAP // 2),
            )
            if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
                game_over = True
            if pipe.off_screen():
                pipes.remove(pipe)
            elif not pipe.passed and pipe.x + PIPE_WIDTH < bird.x:
                pipe.passed = True
                score += 1
                if score > best_score:
                    best_score = score

        bird.draw(screen)

    else:
        # Game over text
        game_over_text = font.render(
            "Game Over! Score: " + str(score) + " Best: " + str(best_score),
            True,
            (0, 0, 0),
        )
        screen.blit(
            game_over_text,
            (
                SCREEN_WIDTH // 2 - game_over_text.get_width() // 2,
                SCREEN_HEIGHT // 2 - 20,
            ),
        )

    # Score display
    score_text = font.render(f"Score: {score}", True, (0, 0, 0))
    screen.blit(score_text, (SCREEN_WIDTH - 150, 20))

    pygame.display.update()
    clock.tick(30)

pygame.quit()
