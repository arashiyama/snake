import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
import pygame
import random
import os
import datetime
import argparse
import threading
import pyttsx3
import json
import sys
from collections import deque

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DQNAgent:
    def __init__(self, state_shape, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000, batch_size=32):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=self.state_shape))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, explore=True):
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def get_q_values(self, state):
        state = np.expand_dims(state, axis=0)
        return self.model.predict(state, verbose=0)[0]
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.zeros((batch_size,) + self.state_shape)
        next_states = np.zeros((batch_size,) + self.state_shape)
        
        for i, (state, _, _, next_state, _) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
        
        targets = self.model.predict(states, verbose=0)
        next_targets = self.target_model.predict(next_states, verbose=0)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.gamma * np.max(next_targets[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, custom_name=None):
        os.makedirs('saved_models', exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = custom_name if custom_name else f"snake_model_{timestamp}.h5"
        filepath = os.path.join('saved_models', filename)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath):
        if os.path.exists(filepath):
            self.model.load_weights(filepath)
            self.update_target_model()
            print(f"Model loaded from {filepath}")
            return True
        return False

class SnakeEnv:
    def __init__(self, grid_size=20, cell_size=20, fps=10, use_gui=True, use_speech=False):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps
        self.use_gui = use_gui
        self.use_speech = use_speech
        
        # Initialize pygame
        self.pygame_initialized = False
        if self.use_gui:
            try:
                pygame.init()
                self.pygame_initialized = True
                self.screen = pygame.display.set_mode((grid_size * cell_size, grid_size * cell_size))
                pygame.display.set_caption("Snake Game")
                self.font = pygame.font.Font(None, 24)
                self.clock = pygame.time.Clock()
            except Exception as e:
                print(f"Pygame initialization error: {e}")
                self.use_gui = False
        
        # Initialize TTS
        self.tts_engine = None
        if self.use_speech:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
            except Exception as e:
                print(f"TTS initialization error: {e}")
                self.use_speech = False
        
        # Game parameters
        self.trap_probability = 0.1
        self.reward_fruit = 1.0
        self.reward_trap = -1.0
        self.reward_collision = -1.0
        self.reward_step = -0.01
        
        # Reset the game
        self.reset()
    
    def update_parameters(self, parameters):
        self.trap_probability = parameters.get("trap_probability", self.trap_probability)
        self.reward_fruit = parameters.get("reward_fruit", self.reward_fruit)
        self.reward_trap = parameters.get("reward_trap", self.reward_trap)
        self.reward_collision = parameters.get("reward_collision", self.reward_collision)
        self.reward_step = parameters.get("reward_step", self.reward_step)
    
    def reset(self):
        # Initialize snake in the middle of the grid
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        
        # Random initial direction
        self.direction = random.choice([(0, -1), (1, 0), (0, 1), (-1, 0)])  # Up, Right, Down, Left
        
        # Place fruit
        self.fruit = self._place_item()
        
        # Clear traps
        self.traps = []
        
        # Add initial traps
        num_traps = int(self.grid_size * self.trap_probability)
        for _ in range(num_traps):
            trap = self._place_item()
            if trap:
                self.traps.append(trap)
        
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        return self._get_state()
    
    def _place_item(self):
        empty_cells = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.snake and (x, y) != self.fruit and (x, y) not in self.traps:
                    empty_cells.append((x, y))
        
        if empty_cells:
            return random.choice(empty_cells)
        return None
    
    def _get_state(self):
        state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        # Snake body
        for segment in self.snake:
            if 0 <= segment[0] < self.grid_size and 0 <= segment[1] < self.grid_size:
                state[segment[1], segment[0], 0] = 1
        
        # Snake head
        head = self.snake[0]
        if 0 <= head[0] < self.grid_size and 0 <= head[1] < self.grid_size:
            state[head[1], head[0], 0] = 2
        
        # Fruit
        if self.fruit and 0 <= self.fruit[0] < self.grid_size and 0 <= self.fruit[1] < self.grid_size:
            state[self.fruit[1], self.fruit[0], 1] = 1
        
        # Traps
        for trap in self.traps:
            if 0 <= trap[0] < self.grid_size and 0 <= trap[1] < self.grid_size:
                state[trap[1], trap[0], 2] = 1
        
        return state
    
    def step(self, action):
        # Map action to direction
        # 0: Up, 1: Right, 2: Down, 3: Left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        # Prevent 180-degree turns
        head_x, head_y = self.snake[0]
        current_dir_x, current_dir_y = self.direction
        
        new_direction = directions[action]
        new_dir_x, new_dir_y = new_direction
        
        # If trying to go in the opposite direction, keep current direction
        if (new_dir_x, new_dir_y) == (-current_dir_x, -current_dir_y) and len(self.snake) > 1:
            new_direction = self.direction
        
        self.direction = new_direction
        
        # Move snake
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)
        
        # Check if out of bounds
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            # Wrap around the grid
            new_head = (new_head[0] % self.grid_size, new_head[1] % self.grid_size)
        
        # Check for collision with self
        if new_head in self.snake:
            self.game_over = True
            reward = self.reward_collision
            if self.use_speech and self.tts_engine:
                self._speak("Ouch! I ran into myself!")
        # Check for collision with trap
        elif new_head in self.traps:
            self.game_over = True
            reward = self.reward_trap
            if self.use_speech and self.tts_engine:
                self._speak("Argh! That was a trap!")
        # Check for fruit
        elif new_head == self.fruit:
            self.snake.insert(0, new_head)
            self.score += 1
            reward = self.reward_fruit
            if self.use_speech and self.tts_engine:
                self._speak("Yum! That was delicious!")
            
            # Place new fruit
            self.fruit = self._place_item()
            
            # Maybe add a new trap
            if random.random() < self.trap_probability:
                trap = self._place_item()
                if trap:
                    self.traps.append(trap)
        else:
            # Just moving
            self.snake.insert(0, new_head)
            self.snake.pop()
            reward = self.reward_step
        
        self.steps += 1
        
        # Check for game over due to too many steps
        if self.steps > 100 * self.grid_size and not self.game_over:
            self.game_over = True
            reward = self.reward_collision
            if self.use_speech and self.tts_engine:
                self._speak("I'm getting tired... time for a nap.")
        
        return self._get_state(), reward, self.game_over, {"score": self.score}
    
    def _speak(self, text):
        try:
            def speak_thread():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            thread = threading.Thread(target=speak_thread)
            thread.daemon = True
            thread.start()
        except Exception as e:
            print(f"Speech error: {e}")
    
    def render(self):
        if not self.use_gui or not self.pygame_initialized:
            return
            
        try:
            self.screen.fill((0, 0, 0))
            
            # Draw snake
            for segment in self.snake:
                pygame.draw.rect(self.screen, (0, 255, 0), 
                                (segment[0] * self.cell_size, segment[1] * self.cell_size, 
                                 self.cell_size, self.cell_size))
            
            # Draw fruit
            if self.fruit:
                pygame.draw.rect(self.screen, (255, 0, 0), 
                                (self.fruit[0] * self.cell_size, self.fruit[1] * self.cell_size, 
                                 self.cell_size, self.cell_size))
            
            # Draw traps
            for trap in self.traps:
                pygame.draw.rect(self.screen, (128, 0, 128), 
                                (trap[0] * self.cell_size, trap[1] * self.cell_size, 
                                 self.cell_size, self.cell_size))
            
            # Draw score
            score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
            self.screen.blit(score_text, (10, 10))
            
            pygame.display.flip()
            self.clock.tick(self.fps)
        except Exception as e:
            print(f"Render error: {e}")
            self.use_gui = False
    
    def close(self):
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False

class GameModifier:
    def __init__(self, file_path="snake_self_awareness.json"):
        self.file_path = file_path
        self.performance_history = []
        self.parameter_changes = []
        self.current_parameters = {
            "trap_probability": 0.1,
            "reward_fruit": 1.0,
            "reward_trap": -1.0,
            "reward_collision": -1.0,
            "reward_step": -0.01
        }
        self.load_state()
    
    def load_state(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    self.performance_history = data.get('performance_history', [])
                    self.parameter_changes = data.get('parameter_changes', [])
                    self.current_parameters = data.get('current_parameters', self.current_parameters)
                print(f"Loaded self-awareness data from {self.file_path}")
            except Exception as e:
                print(f"Error loading self-awareness data: {e}")
    
    def save_state(self):
        data = {
            'performance_history': self.performance_history,
            'parameter_changes': self.parameter_changes,
            'current_parameters': self.current_parameters
        }
        try:
            with open(self.file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved self-awareness data to {self.file_path}")
        except Exception as e:
            print(f"Error saving self-awareness data: {e}")
    
    def log_performance(self, episode, score, steps, avg_reward):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        performance = {
            'timestamp': timestamp,
            'episode': episode,
            'score': score,
            'steps': steps,
            'avg_reward': avg_reward,
            'parameters': self.current_parameters.copy()
        }
        self.performance_history.append(performance)
        
        # Every 100 episodes, consider modifying parameters
        if episode > 0 and episode % 100 == 0:
            self.consider_parameter_changes(episode)
            self.save_state()
    
    def consider_parameter_changes(self, episode):
        if len(self.performance_history) < 100:
            return  # Not enough data yet
        
        # Get recent performance
        recent = self.performance_history[-100:]
        avg_score = sum(p['score'] for p in recent) / len(recent)
        avg_steps = sum(p['steps'] for p in recent) / len(recent)
        avg_reward = sum(p['avg_reward'] for p in recent) / len(recent)
        
        # Get previous performance if available
        if len(self.performance_history) >= 200:
            previous = self.performance_history[-200:-100]
            prev_avg_score = sum(p['score'] for p in previous) / len(previous)
            prev_avg_steps = sum(p['steps'] for p in previous) / len(previous)
            prev_avg_reward = sum(p['avg_reward'] for p in previous) / len(previous)
            
            score_change = avg_score - prev_avg_score
            steps_change = avg_steps - prev_avg_steps
            reward_change = avg_reward - prev_avg_reward
        else:
            # No previous data, assume neutral change
            score_change = 0
            steps_change = 0
            reward_change = 0
        
        # Decide if we should make changes
        should_change = False
        change_reason = ""
        
        # If performance is stagnant or declining
        if score_change <= 0 and reward_change <= 0:
            should_change = True
            change_reason = "Performance stagnant or declining"
        
        # If we're doing too well, make it harder
        elif score_change > 5 and avg_score > 20:
            should_change = True
            change_reason = "Performance too good, increasing difficulty"
        
        if should_change:
            self.modify_parameters(episode, change_reason, score_change, reward_change)
    
    def modify_parameters(self, episode, reason, score_change, reward_change):
        old_params = self.current_parameters.copy()
        
        # If performance is declining, make it easier
        if score_change < 0:
            # Decrease trap probability
            self.current_parameters["trap_probability"] = max(0.05, 
                                                            self.current_parameters["trap_probability"] - 0.02)
            # Increase fruit reward
            self.current_parameters["reward_fruit"] = min(1.5, 
                                                        self.current_parameters["reward_fruit"] + 0.1)
            # Decrease penalties
            self.current_parameters["reward_trap"] = min(-0.5, 
                                                        self.current_parameters["reward_trap"] + 0.1)
            self.current_parameters["reward_collision"] = min(-0.5, 
                                                            self.current_parameters["reward_collision"] + 0.1)
            self.current_parameters["reward_step"] = min(-0.005, 
                                                        self.current_parameters["reward_step"] + 0.005)
        
        # If performance is too good, make it harder
        elif score_change > 5:
            # Increase trap probability
            self.current_parameters["trap_probability"] = min(0.2, 
                                                            self.current_parameters["trap_probability"] + 0.02)
            # Decrease fruit reward
            self.current_parameters["reward_fruit"] = max(0.8, 
                                                        self.current_parameters["reward_fruit"] - 0.1)
            # Increase penalties
            self.current_parameters["reward_trap"] = max(-1.5, 
                                                        self.current_parameters["reward_trap"] - 0.1)
            self.current_parameters["reward_collision"] = max(-1.5, 
                                                            self.current_parameters["reward_collision"] - 0.1)
            self.current_parameters["reward_step"] = max(-0.02, 
                                                        self.current_parameters["reward_step"] - 0.005)
        
        # Random exploration of parameters occasionally
        elif random.random() < 0.3:
            param_to_change = random.choice(list(self.current_parameters.keys()))
            if param_to_change == "trap_probability":
                self.current_parameters[param_to_change] = max(0.05, min(0.2, 
                                                                        self.current_parameters[param_to_change] + 
                                                                        random.uniform(-0.03, 0.03)))
            elif param_to_change in ["reward_fruit"]:
                self.current_parameters[param_to_change] = max(0.5, min(1.5, 
                                                                        self.current_parameters[param_to_change] + 
                                                                        random.uniform(-0.2, 0.2)))
            elif param_to_change in ["reward_trap", "reward_collision"]:
                self.current_parameters[param_to_change] = max(-1.5, min(-0.5, 
                                                                        self.current_parameters[param_to_change] + 
                                                                        random.uniform(-0.2, 0.2)))
            elif param_to_change == "reward_step":
                self.current_parameters[param_to_change] = max(-0.02, min(-0.005, 
                                                                        self.current_parameters[param_to_change] + 
                                                                        random.uniform(-0.005, 0.005)))
            reason += f" (random exploration of {param_to_change})"
        
        # Log the change
        change = {
            'episode': episode,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'reason': reason,
            'old_parameters': old_params,
            'new_parameters': self.current_parameters.copy()
        }
        self.parameter_changes.append(change)
        
        print(f"\nSnake self-awareness update (Episode {episode}):")
        print(f"Reason: {reason}")
        print("Parameter changes:")
        for param in self.current_parameters:
            if old_params[param] != self.current_parameters[param]:
                print(f"  {param}: {old_params[param]:.4f} -> {self.current_parameters[param]:.4f}")
        print()
    
    def get_parameters(self):
        return self.current_parameters

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a DQN agent to play Snake')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--grid-size', type=int, default=30, help='Size of the grid')
    parser.add_argument('--cell-size', type=int, default=15, help='Size of each cell in pixels')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--load-model', type=str, help='Path to model to load')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI')
    parser.add_argument('--no-speech', action='store_true', help='Disable speech')
    parser.add_argument('--self-aware', action='store_true', help='Enable self-awareness')
    parser.add_argument('--render', action='store_true', help='Force rendering')
    args = parser.parse_args()
    
    # Initialize environment
    grid_size = args.grid_size
    cell_size = args.cell_size
    env = SnakeEnv(grid_size=grid_size, cell_size=cell_size, fps=args.fps, 
                  use_gui=not args.no_gui, use_speech=not args.no_speech)
    
    # Get state and action space info
    state = env.reset()
    state_shape = state.shape
    action_size = 4  # Up, Right, Down, Left
    
    # Initialize game modifier for self-awareness
    game_modifier = None
    if args.self_aware:
        game_modifier = GameModifier()
        env.update_parameters(game_modifier.get_parameters())
    
    # Initialize agent
    agent = DQNAgent(state_shape, action_size)
    
    # Load model if specified
    if args.load_model:
        agent.load_model(args.load_model)
        # Start with lower epsilon for pretrained model
        agent.epsilon = max(0.1, agent.epsilon * 0.1)
    
    # Training loop
    episodes = args.episodes
    batch_size = 32
    update_target_every = 5
    
    total_rewards = []
    
    try:
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            
            while True:
                # Process pygame events
                if env.pygame_initialized:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\nTraining interrupted by user (window closed)")
                            raise KeyboardInterrupt
                
                # Get action
                action = agent.act(state)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Remember experience
                agent.remember(state, action, reward, next_state, done)
                
                # Update state and reward
                state = next_state
                total_reward += reward
                
                # Render
                if args.render or not args.no_gui:
                    env.render()
                
                if done:
                    # Update target network periodically
                    if e % update_target_every == 0:
                        agent.update_target_model()
                    
                    # Log performance
                    score = info.get('score', 0)
                    print(f"Episode: {e+1}/{episodes}, Score: {score}, " +
                          f"Steps: {env.steps}, Reward: {total_reward:.2f}, " +
                          f"Epsilon: {agent.epsilon:.4f}")
                    
                    # Log to game modifier
                    if game_modifier:
                        game_modifier.log_performance(e, score, env.steps, total_reward)
                        env.update_parameters(game_modifier.get_parameters())
                    
                    total_rewards.append(total_reward)
                    break
            
            # Train the agent
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining stopped due to error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save final model
        agent.save_model()
        if game_modifier:
            game_modifier.save_state()
        env.close()
        
        # Plot training results if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(total_rewards)
            plt.title('Training Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.savefig('training_rewards.png')
            print("Training rewards plot saved to training_rewards.png")
        except ImportError:
            print("Matplotlib not available, skipping plot generation")
        except Exception as e:
            print(f"Error generating plot: {e}")

if __name__ == "__main__":
    main()