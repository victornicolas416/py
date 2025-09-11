import pygame
import sys
import random
import math
import os
import json
from copy import deepcopy

# ================= CONFIGURATION ================= #
WIDTH, HEIGHT = 400, 600
FPS = 60
GRAVITY = 0.5
FLAP_STRENGTH = -8
PIPE_WIDTH = 70
PIPE_GAP = 150
PIPE_SPEED = 3
BIRD_RADIUS = 10

BG_COLOR = (135, 206, 235)
BIRD_COLOR = (255, 255, 0)
PIPE_COLOR = (0, 200, 0)

BEST_GA_FILE = "best_ga.json"

# ================= UTILITIES ================= #
def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

# ================= CLASSES JEU ================= #
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

# ================= RESEAU FEEDFORWARD ================= #
class NeuralNet:
    def __init__(self, hidden_size=6):
        self.input_size = 3
        self.hidden_size = hidden_size
        self.w1 = [[random.uniform(-1,1) for _ in range(self.input_size)] for _ in range(self.hidden_size)]
        self.b1 = [0.0 for _ in range(self.hidden_size)]
        self.w2 = [random.uniform(-1,1) for _ in range(self.hidden_size)]
        self.b2 = 0.0

    def predict(self, inputs):
        h = []
        for i in range(self.hidden_size):
            s = sum(w*x for w,x in zip(self.w1[i], inputs)) + self.b1[i]
            h.append(math.tanh(s))
        out = sum(w*h_i for w,h_i in zip(self.w2, h)) + self.b2
        return sigmoid(out)

    def clone(self):
        return deepcopy(self)

    def mutate(self, rate=0.1, scale=0.5):
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                if random.random() < rate:
                    self.w1[i][j] += random.uniform(-scale, scale)
            if random.random() < rate:
                self.b1[i] += random.uniform(-scale, scale)
        for i in range(self.hidden_size):
            if random.random() < rate:
                self.w2[i] += random.uniform(-scale, scale)
        if random.random() < rate:
            self.b2 += random.uniform(-scale, scale)

    @staticmethod
    def crossover(a,b):
        child = NeuralNet(hidden_size=a.hidden_size)
        for i in range(a.hidden_size):
            for j in range(a.input_size):
                child.w1[i][j] = (a.w1[i][j]+b.w1[i][j])/2.0
            child.b1[i] = (a.b1[i]+b.b1[i])/2.0
            child.w2[i] = (a.w2[i]+b.w2[i])/2.0
        child.b2 = (a.b2+b.b2)/2.0
        return child

    def to_dict(self):
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2, "hidden": self.hidden_size}

    @staticmethod
    def from_dict(d):
        n = NeuralNet(hidden_size=d.get("hidden",6))
        n.w1 = d["w1"]
        n.b1 = d["b1"]
        n.w2 = d["w2"]
        n.b2 = d["b2"]
        return n

# ================= FONCTIONS ================= #
def spawn_pipe():
    gap_y = random.randint(100, HEIGHT-200)
    return Pipe(WIDTH, gap_y)

def check_collision(bird, pipes):
    b_rect = bird.rect()
    if bird.y - BIRD_RADIUS <=0 or bird.y + BIRD_RADIUS >= HEIGHT:
        return True
    for p in pipes:
        top_rect = pygame.Rect(p.x, 0, PIPE_WIDTH, p.gap_y - PIPE_GAP/2)
        bottom_rect = pygame.Rect(p.x, p.gap_y + PIPE_GAP/2, PIPE_WIDTH, HEIGHT-(p.gap_y + PIPE_GAP/2))
        if b_rect.colliderect(top_rect) or b_rect.colliderect(bottom_rect):
            return True
    return False

def get_inputs(bird, pipes):
    if not pipes:
        return [bird.y/HEIGHT,0.0,0.0]
    next_pipe = min([p for p in pipes if p.x + PIPE_WIDTH > bird.x], key=lambda p:p.x, default=None)
    if not next_pipe:
        return [bird.y/HEIGHT,0.0,0.0]
    return [bird.y/HEIGHT,(next_pipe.gap_y - bird.y)/HEIGHT,(next_pipe.x - bird.x)/WIDTH]

# ================= ALGO GENETIQUE ================= #
class GA:
    def __init__(self, pop_size=40, hidden=8, elite_frac=0.2, mutate_rate=0.12, mutate_scale=0.6):
        self.pop_size = pop_size
        self.hidden = hidden
        self.elite_frac = elite_frac
        self.mutate_rate = mutate_rate
        self.mutate_scale = mutate_scale
        self.population = [NeuralNet(hidden_size=self.hidden) for _ in range(pop_size)]
        self.best = None
        self.best_score = 0
        self.generation = 0
        if os.path.exists(BEST_GA_FILE):
            try:
                with open(BEST_GA_FILE,"r") as f:
                    self.best = NeuralNet.from_dict(json.load(f))
            except:
                pass

    def evaluate_and_evolve(self, agents_fitness_and_nets):
        agents = sorted(agents_fitness_and_nets,key=lambda x:x[0],reverse=True)
        if agents and agents[0][0]>self.best_score:
            self.best_score = agents[0][0]
            self.best = agents[0][1].clone()
            try:
                with open(BEST_GA_FILE,"w") as f:
                    json.dump(self.best.to_dict(), f)
            except:
                pass
        elite_count = max(1,int(self.pop_size*self.elite_frac))
        elites = [deepcopy(net) for _,net in agents[:elite_count]]
        new_pop = elites[:]
        while len(new_pop)<self.pop_size:
            a=random.choice(agents)[1]
            b=random.choice(agents)[1]
            child = NeuralNet.crossover(a,b)
            child.mutate(rate=self.mutate_rate,scale=self.mutate_scale)
            new_pop.append(child)
        self.population = new_pop
        self.generation += 1

# ================= INITIALISATION PYGAME ================= #
pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Flappy Bird - GA IA")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None,30)
font_big = pygame.font.SysFont(None,48)

# variables globales
bird = Bird(80, HEIGHT//2)
pipes = []
SPAWNPIPE = pygame.USEREVENT + 1
pygame.time.set_timer(SPAWNPIPE,1500)
score = 0

# GA instance
ga = GA(pop_size=36,hidden=8)

# ================= HELPERS ================= #
def draw_text_center(text,y,size_font=font,color=(255,255,255)):
    surf = size_font.render(text,True,color)
    screen.blit(surf,(WIDTH//2 - surf.get_width()//2, y))

def reset_single():
    return Bird(80, HEIGHT//2), [], 0

def init_ga_run():
    agents = []
    nets = ga.population[:]
    birds = [Bird(80, HEIGHT//2) for _ in range(len(nets))]
    scores = [0]*len(nets)
    alive = [True]*len(nets)
    time_alive = [0]*len(nets)
    return nets,birds,scores,alive,time_alive

# ================= BOUCLE + MENU ================= #
mode = "menu"
running=True
bird,pipes,score = reset_single()

while running:
    if mode=="menu":
        screen.fill((50,50,50))
        draw_text_center("Flappy Bird - Menu",60,font_big)
        draw_text_center("1 - Mode Normal",160)
        draw_text_center("2 - Mode Entrainement",200)
        draw_text_center("3 - Mode IA Auto",240)
        draw_text_center("4 - GA Training",280)
        draw_text_center("ECHAP - Quitter",320)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_1:
                    mode="normal"
                    bird,pipes,score = reset_single()
                if event.key==pygame.K_2:
                    mode="train"
                    bird,pipes,score = reset_single()
                if event.key==pygame.K_3:
                    mode="ai"
                    bird,pipes,score = reset_single()
                if event.key==pygame.K_4:
                    mode="ga"
                    nets,birds,scores,alive,time_alive = init_ga_run()
                    pipes=[]
                    score=0
                if event.key==pygame.K_ESCAPE:
                    running=False
        continue

    # events
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running=False
        if event.type==pygame.KEYDOWN:
            if event.key==pygame.K_ESCAPE:
                mode="menu"
                bird,pipes,score=reset_single()
        if event.type==SPAWNPIPE:
            pipes.append(spawn_pipe())

    # --- Mode Normal / Train / IA Auto ---
    if mode in ["normal","train","ai"]:
        if mode=="ai" and ga.best:
            inputs = get_inputs(bird,pipes)
            if ga.best.predict(inputs)>0.5:
                bird.flap()
        if mode=="train":
            # optionnel: collecte data pour futur apprentissage supervisé
            pass

        bird.update()
        for p in pipes: p.update()
        pipes = [p for p in pipes if p.x + PIPE_WIDTH >0]

        if check_collision(bird,pipes):
            mode="menu"
            bird,pipes,score=reset_single()
            continue

        for p in pipes:
            if not p.passed and p.x + PIPE_WIDTH < bird.x:
                p.passed=True
                score+=1

        # affichage
        screen.fill(BG_COLOR)
        pygame.draw.circle(screen,BIRD_COLOR,(int(bird.x),int(bird.y)),BIRD_RADIUS)
        for p in pipes:
            pygame.draw.rect(screen,PIPE_COLOR,(p.x,0,PIPE_WIDTH,p.gap_y - PIPE_GAP/2))
            pygame.draw.rect(screen,PIPE_COLOR,(p.x,p.gap_y + PIPE_GAP/2,PIPE_WIDTH,HEIGHT-(p.gap_y + PIPE_GAP/2)))
        draw_text_center(f"Score: {score}",10)
        pygame.display.flip()
        clock.tick(FPS)

    # --- Mode GA Training ---
    if mode=="ga":
        alive_count = sum(alive)
        for i,b in enumerate(birds):
            if alive[i]:
                inputs = get_inputs(b,pipes)
                if nets[i].predict(inputs)>0.5:
                    b.flap()
                b.update()
                for p in pipes: p.update()
                pipes = [p for p in pipes if p.x + PIPE_WIDTH>0]
                if check_collision(b,pipes):
                    alive[i]=False
        for i,b in enumerate(birds):
            for p in pipes:
                if not p.passed and p.x + PIPE_WIDTH < b.x and alive[i]:
                    p.passed=True
                    scores[i]+=1
        time_alive = [t+1 for t,t_alive in zip(time_alive,alive) if t_alive]

        # fin de génération
        if all(not a for a in alive):
            fitness_nets = [(s,net) for s,net in zip(scores,nets)]
            ga.evaluate_and_evolve(fitness_nets)
            nets,birds,scores,alive,time_alive = init_ga_run()
            pipes=[]
            score=0

        # affichage GA
        screen.fill(BG_COLOR)
        for b in birds:
            if b.y>0 and b.y<HEIGHT:
                pygame.draw.circle(screen,BIRD_COLOR,(int(b.x),int(b.y)),BIRD_RADIUS)
        for p in pipes:
            pygame.draw.rect(screen,PIPE_COLOR,(p.x,0,PIPE_WIDTH,p.gap_y - PIPE_GAP/2))
            pygame.draw.rect(screen,PIPE_COLOR,(p.x,p.gap_y + PIPE_GAP/2,PIPE_WIDTH,HEIGHT-(p.gap_y + PIPE_GAP/2)))
        draw_text_center(f"Generation: {ga.generation}",10)
        draw_text_center(f"Best score: {ga.best_score}",40)
        draw_text_center(f"Alive: {alive_count}",70)
        pygame.display.flip()
        clock.tick(FPS)
