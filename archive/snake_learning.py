import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Environment: SnakeEnv
# ---------------------------
class SnakeEnv:
    def __init__(self, grid_width=20, grid_height=20, max_steps=200):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        # Start with a one-segment snake in the center and default direction to right.
        self.snake = [(self.grid_width // 2, self.grid_height // 2)]
        self.direction = (1, 0)  # right
        self.fruit = self._get_random_fruit()
        self.traps = []
        self.score = 0
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        """
        Returns a grid (2D NumPy array) of shape (grid_height, grid_width) where:
          - 0 indicates an empty cell,
          - 1 indicates a cell occupied by the snake,
          - 2 indicates the fruit,
          - 3 indicates a trap.
        """
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        for (x, y) in self.snake:
            grid[y, x] = 1
        fx, fy = self.fruit
        grid[fy, fx] = 2
        for (x, y) in self.traps:
            grid[y, x] = 3
        return grid

    def step(self, action):
        """
        Takes an action (0: up, 1: right, 2: down, 3: left) and updates the game.
        Returns: next_state, reward, done, info
        """
        # Map action to movement vector.
        mapping = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        desired = mapping[action]
        # Prevent the snake from reversing direction if its length > 1.
        if len(self.snake) > 1 and (desired[0] == -self.direction[0] and desired[1] == -self.direction[1]):
            move = self.direction
        else:
            move = desired
            self.direction = move

        new_head = (self.snake[0][0] + move[0], self.snake[0][1] + move[1])
        self.steps += 1
        reward = 0
        done = False

        # Check for collision with walls.
        if (new_head[0] < 0 or new_head[0] >= self.grid_width or 
            new_head[1] < 0 or new_head[1] >= self.grid_height):
            reward = -10
            done = True
            return self._get_state(), reward, done, {}

        # Check for collision with itself.
        if new_head in self.snake:
            reward = -10
            done = True
            return self._get_state(), reward, done, {}

        # Check for collision with a trap.
        if new_head in self.traps:
            self.traps.remove(new_head)
            # Lose half the snake’s length (rounded down).
            penalty = len(self.snake) // 2
            self.snake = self.snake[:-penalty] if len(self.snake) > penalty else []
            reward = -5
            if len(self.snake) == 0:
                done = True
                return self._get_state(), reward, done, {}

        # Move the snake.
        self.snake.insert(0, new_head)
        if new_head == self.fruit:
            reward = 10
            self.score += 10
            self.fruit = self._get_random_fruit()
        else:
            self.snake.pop()

        # Add traps with some probability.
        if random.random() < 0.5:
            for _ in range(2):
                trap_pos = self._get_random_trap()
                if trap_pos is not None:
                    self.traps.append(trap_pos)

        if self.steps >= self.max_steps:
            done = True
        return self._get_state(), reward, done, {}

    def _get_random_fruit(self):
        while True:
            pos = (random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1))
            if pos not in self.snake:
                return pos

    def _get_random_trap(self):
        for _ in range(100):
            pos = (random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1))
            if pos not in self.snake and pos != self.fruit and pos not in self.traps:
                return pos
        return None

# ---------------------------
# DQN Agent
# ---------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # e.g. 20*20 = 400
        self.action_size = action_size  # 4 possible actions
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount factor
        self.epsilon = 1.0   # exploration rate (will decay over time)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Build a simple feed-forward network.
        model = Sequential()
        model.add(Flatten(input_shape=(self.state_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # ε-greedy action selection.
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ---------------------------
# Main Training Loop
# ---------------------------
if __name__ == "__main__":
    env = SnakeEnv(grid_width=20, grid_height=20, max_steps=200)
    initial_state = env.reset()
    # Flatten the grid to a vector.
    state_size = initial_state.shape[0] * initial_state.shape[1]
    action_size = 4  # up, right, down, left
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = env.reset().flatten()
        total_reward = 0
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print("Episode: {}/{}, Score: {}, Total Reward: {}, Epsilon: {:.2f}"
                      .format(e, episodes, env.score, total_reward, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
