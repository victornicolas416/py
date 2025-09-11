import pygame
import sys
import random

# ================= CONFIGURATION ================= #
WIDTH, HEIGHT = 400, 600
FPS = 60
GRAVITY = 0.5
FLAP_STRENGTH = -8
PIPE_WIDTH = 70
PIPE_GAP = 150
PIPE_SPEED = 3
BIRD_RADIUS = 15

BG_COLOR = (135, 206, 235)
BIRD_COLOR = (255, 255, 0)
PIPE_COLOR = (0, 200, 0)

# ================= CLASSES ================= #
class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vy = 0

    def flap(self):
        self.vy = FLAP_STRENGTH

    def update(self):
        self.vy += GRAVITY
        self.y += self.vy

    def rect(self):
        return pygame.Rect(self.x - BIRD_RADIUS, self.y - BIRD_RADIUS, BIRD_RADIUS*2, BIRD_RADIUS*2)

class Pipe:
    def __init__(self, x, gap_y):
        self.x = x
        self.gap_y = gap_y
        self.passed = False

    def update(self):
        self.x -= PIPE_SPEED

# ================= FONCTIONS ================= #
def spawn_pipe():
    gap_y = random.randint(100, HEIGHT - 200)
    return Pipe(WIDTH, gap_y)

def check_collision(bird, pipes):
    b_rect = bird.rect()
    if bird.y - BIRD_RADIUS <= 0 or bird.y + BIRD_RADIUS >= HEIGHT:
        return True
    for p in pipes:
        top_rect = pygame.Rect(p.x, 0, PIPE_WIDTH, p.gap_y - PIPE_GAP/2)
        bottom_rect = pygame.Rect(p.x, p.gap_y + PIPE_GAP/2, PIPE_WIDTH, HEIGHT - (p.gap_y + PIPE_GAP/2))
        if b_rect.colliderect(top_rect) or b_rect.colliderect(bottom_rect):
            return True
    return False

# ================= INITIALISATION ================= #
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird Basique")
clock = pygame.time.Clock()

bird = Bird(80, HEIGHT//2)
pipes = []
SPAWNPIPE = pygame.USEREVENT + 1
pygame.time.set_timer(SPAWNPIPE, 1500)
score = 0
running = True

font = pygame.font.SysFont(None, 48)

# ================= BOUCLE DE JEU ================= #
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                bird.flap()
            if event.key == pygame.K_r:
                bird = Bird(80, HEIGHT//2)
                pipes = []
                score = 0
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
        if event.type == SPAWNPIPE:
            pipes.append(spawn_pipe())

    # Mise à jour
    bird.update()
    for p in pipes:
        p.update()
    pipes = [p for p in pipes if p.x + PIPE_WIDTH > 0]

    # Vérification collision
    if check_collision(bird, pipes):
        bird = Bird(80, HEIGHT//2)
        pipes = []
        score = 0

    # Mise à jour du score
    for p in pipes:
        if not p.passed and p.x + PIPE_WIDTH < bird.x:
            p.passed = True
            score += 1

    # Affichage
    screen.fill(BG_COLOR)
    pygame.draw.circle(screen, BIRD_COLOR, (int(bird.x), int(bird.y)), BIRD_RADIUS)
    for p in pipes:
        pygame.draw.rect(screen, PIPE_COLOR, (p.x, 0, PIPE_WIDTH, p.gap_y - PIPE_GAP/2))
        pygame.draw.rect(screen, PIPE_COLOR, (p.x, p.gap_y + PIPE_GAP/2, PIPE_WIDTH, HEIGHT - (p.gap_y + PIPE_GAP/2)))

    score_text = font.render(str(score), True, (255,255,255))
    screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, 20))

    pygame.display.flip()
    clock.tick(FPS)
