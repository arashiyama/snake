import pygame
import random
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --------------------------
# Game Settings and Helpers
# --------------------------
BLOCK_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
SCREEN_WIDTH = GRID_WIDTH * BLOCK_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * BLOCK_SIZE

# Colors (R, G, B)
WHITE = (255, 255, 255)
BLACK = (  0,   0,   0)
RED   = (255,   0,   0)
GREEN = (0,   255,   0)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# For replay memory
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Add these constants after the existing Game Settings
POINTS_FRUIT = 100        # Points for eating fruit
POINTS_SURVIVAL = 1       # Points per frame
POINTS_DEATH = -100       # Points for dying
LENGTH_BONUS_THRESHOLD = 5  # Snake length needed for bonus
LENGTH_BONUS_POINTS = 500   # Bonus points for reaching threshold

# --------------------------
# Snake Game Environment
# --------------------------
class SnakeGameEnv:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("RL Snake")
            self.clock = pygame.time.Clock()
        self.length_bonus_awarded = False
        self.reset()

    def reset(self):
        # Initialize snake in center; snake is a list of (x, y) blocks
        self.direction = RIGHT
        self.head = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
        self.snake = [self.head,
                      (self.head[0] - 1, self.head[1]),
                      (self.head[0] - 2, self.head[1])]
        self.score = 0
        self._place_food()
        self.frame_iteration = 0
        self.length_bonus_awarded = False
        return self.get_state()

    def _place_food(self):
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            self.food = (x, y)
            if self.food not in self.snake:
                break

    def play_step(self, action):
        """
        action: 0 = straight, 1 = right turn, 2 = left turn
        Returns: state, reward, done, score
        """
        self.frame_iteration += 1
        self._move(action)  # update the head
        self.snake.insert(0, self.head)  # move head
        MAX_STEPS = 200  # You can adjust this value

    if self.frame_iteration > MAX_STEPS * len(self.snake):
        # Optionally assign a small negative reward to discourage inaction
        reward = -5
        game_over = True
        return self.get_state(), reward, game_over, self.score


        reward = POINTS_SURVIVAL  # Base reward for surviving
        game_over = False

        # Check if game over (collision with wall or self)
        if self._is_collision():
            game_over = True
            reward = POINTS_DEATH
            return self.get_state(), reward, game_over, self.score

        # Check if food eaten
        if self.head == self.food:
            self.score += POINTS_FRUIT
            reward += POINTS_FRUIT
            
            # Check for length bonus
            if len(self.snake) >= LENGTH_BONUS_THRESHOLD and not self.length_bonus_awarded:
                self.score += LENGTH_BONUS_POINTS
                reward += LENGTH_BONUS_POINTS
                self.length_bonus_awarded = True
                
            self._place_food()
        else:
            self.snake.pop()  # remove tail

        if self.render_mode:
            self._update_ui()
            self.clock.tick(60)

        return self.get_state(), reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Collision with boundaries
        if pt[0] < 0 or pt[0] >= GRID_WIDTH or pt[1] < 0 or pt[1] >= GRID_HEIGHT:
            return True
        # Collision with self
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt[0]*BLOCK_SIZE, pt[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        # Score and Length
        font = pygame.font.SysFont('arial', 25)
        score_text = f"Score: {self.score} | Length: {len(self.snake)}"
        text = font.render(score_text, True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """
        action: 0 = straight, 1 = right, 2 = left
        Updates the direction and head based on the action.
        """
        # Define ordered directions for turning: [UP, RIGHT, DOWN, LEFT]
        directions = [UP, RIGHT, DOWN, LEFT]
        idx = directions.index(self.direction)

        if action == 1:  # right turn: clockwise rotation
            new_dir = directions[(idx + 1) % 4]
        elif action == 2:  # left turn: counter-clockwise rotation
            new_dir = directions[(idx - 1) % 4]
        else:  # action 0: no change
            new_dir = self.direction

        self.direction = new_dir
        x = self.head[0] + self.direction[0]
        y = self.head[1] + self.direction[1]
        self.head = (x, y)

    def get_state(self):
        """
        Returns an 11-dimensional state:
        [danger straight, danger right, danger left,
         moving direction (up, down, left, right) as one-hot (4),
         food location (food left, food right, food up, food down)]
        """
        head = self.snake[0]
        point_l = (head[0] + LEFT[0], head[1] + LEFT[1])
        point_r = (head[0] + RIGHT[0], head[1] + RIGHT[1])
        point_s = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Determine danger in the three directions
        danger_straight = 1 if self._is_collision(point_s) else 0
        danger_right = 1 if self._is_collision(self._next_point(self.direction, 1)) else 0
        danger_left = 1 if self._is_collision(self._next_point(self.direction, -1)) else 0

        # Current direction one-hot
        dir_up = 1 if self.direction == UP else 0
        dir_down = 1 if self.direction == DOWN else 0
        dir_left = 1 if self.direction == LEFT else 0
        dir_right = 1 if self.direction == RIGHT else 0

        # Food location relative to head
        food_left = 1 if self.food[0] < head[0] else 0
        food_right = 1 if self.food[0] > head[0] else 0
        food_up = 1 if self.food[1] < head[1] else 0
        food_down = 1 if self.food[1] > head[1] else 0

        state = [
            danger_straight, danger_right, danger_left,
            dir_up, dir_down, dir_left, dir_right,
            food_left, food_right, food_up, food_down
        ]
        return np.array(state, dtype=int)

    def _next_point(self, current_dir, turn):  
        """
        Computes the next point if we turn relative to current_dir.
        turn: 1 for right, -1 for left.
        """
        directions = [UP, RIGHT, DOWN, LEFT]
        idx = directions.index(self.direction)
        new_dir = directions[(idx + turn) % 4]
        head = self.snake[0]
        return (head[0] + new_dir[0], head[1] + new_dir[1])

    def close(self):
        if self.render_mode:
            pygame.quit()

# --------------------------
# Deep Q-Network (DQN)
# --------------------------
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# --------------------------
# Experience Replay Memory
# --------------------------
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --------------------------
# Helper Functions for DQN Training
# --------------------------
def get_tensor(x):
    return torch.tensor(x, dtype=torch.float)

# Hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# --------------------------
# Agent Class
# --------------------------
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # will be set in train loop
        self.gamma = 0.9  # discount rate
        self.memory = ReplayMemory(MAX_MEMORY)
        self.model = DQN(input_size=11, hidden_size=128, output_size=3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def get_action(self, state):
        # ε-greedy: choose a random action with probability epsilon
        self.epsilon = max(80 - self.n_games, 0)  # Decaying epsilon
        if random.randint(0, 200) < self.epsilon:
            action = random.randint(0, 2)
        else:
            state0 = get_tensor(state)
            prediction = self.model(state0)
            action = torch.argmax(prediction).item()
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            mini_sample = self.memory.sample(len(self.memory))
        else:
            mini_sample = self.memory.sample(BATCH_SIZE)

        states = get_tensor(np.array([t.state for t in mini_sample]))
        actions = torch.tensor([t.action for t in mini_sample], dtype=torch.long)
        rewards = get_tensor(np.array([t.reward for t in mini_sample]))
        next_states = get_tensor(np.array([t.next_state for t in mini_sample]))
        dones = torch.tensor([t.done for t in mini_sample], dtype=torch.bool)

        # Q value for current states
        pred = self.model(states)
        # Gather the Q-values for the actions taken
        pred = pred.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Q value for next states
        target = rewards + self.gamma * torch.max(self.model(next_states), dim=1)[0] * (~dones)
        loss = F.mse_loss(pred, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_short_memory(self, state, action, reward, next_state, done):
        state0 = get_tensor(state).unsqueeze(0)
        next_state0 = get_tensor(next_state).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.long)
        reward_tensor = get_tensor([reward])
        done_tensor = torch.tensor([done], dtype=torch.bool)

        pred = self.model(state0)
        pred = pred.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        target = reward_tensor + self.gamma * torch.max(self.model(next_state0), dim=1)[0] * (not done)
        loss = F.mse_loss(pred, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# --------------------------
# Training Loop
# --------------------------
def train():
    agent = Agent()
    game = SnakeGameEnv(render_mode=False)
    scores = []
    total_score = 0
    n_episodes = 500  # Increase this number for better training

    for episode in range(n_episodes):
        state = game.reset()
        done = False
        score = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, score = game.play_step(action)
            agent.train_short_memory(state, action, reward, next_state, done)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

        agent.n_games += 1
        agent.train_long_memory()

        scores.append(score)
        total_score += score
        mean_score = total_score / (episode + 1)
        print(f"Episode {episode + 1} Score: {score} Mean Score: {mean_score:.2f}")

    game.close()
    # Save the model if desired:
    torch.save(agent.model.state_dict(), "dqn_snake.pth")

if __name__ == '__main__':
    train()
