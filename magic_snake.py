import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
import pygame
import os
import sys
import json
import time
import threading
from datetime import datetime
import pyttsx3
import platform

# ---------------------------
# Game Modifier for Self-Awareness
# ---------------------------
class GameModifier:
    """A class that allows the snake to modify its own game parameters"""
    def __init__(self):
        self.modifications = {
            'trap_probability': 0.2,
            'fruit_value': 10,
            'trap_penalty': 5,
            'wall_penalty': 10,
            'movement_reward': 0.1,
            'max_steps': 200,
            'learning_rate': 0.001,
            'epsilon_decay': 0.998
        }
        self.last_modification_time = 0
        self.modification_cooldown = 10  # seconds between modifications
        self.performance_history = []
        self.modification_history = []
        
    def log_performance(self, score, total_reward, steps):
        """Log the snake's performance for decision making"""
        self.performance_history.append({
            'score': score,
            'total_reward': total_reward,
            'steps': steps,
            'time': time.time()
        })
        # Keep only the last 50 episodes
        if len(self.performance_history) > 50:
            self.performance_history.pop(0)
    
    def decide_modification(self):
        """Decide if and how to modify the game based on performance"""
        # Need enough history to make decisions
        if len(self.performance_history) < 10:
            return None, None
            
        # Check cooldown
        if time.time() - self.last_modification_time < self.modification_cooldown:
            return None, None
            
        # Analyze recent performance
        recent = self.performance_history[-10:]
        older = self.performance_history[-20:-10] if len(self.performance_history) >= 20 else []
        
        avg_recent_score = sum(p['score'] for p in recent) / len(recent)
        avg_recent_reward = sum(p['total_reward'] for p in recent) / len(recent)
        
        # If we have older data, compare to see if we're improving
        improving = False
        if older:
            avg_older_score = sum(p['score'] for p in older) / len(older)
            avg_older_reward = sum(p['total_reward'] for p in older) / len(older)
            improving = avg_recent_score > avg_older_score and avg_recent_reward > avg_older_reward
        
        # Make decisions based on performance
        modification = None
        reason = None
        
        # If scores are very low, make the game easier
        if avg_recent_score < 5:
            if self.modifications['trap_probability'] > 0.05:
                modification = ('trap_probability', max(0.05, self.modifications['trap_probability'] - 0.05))
                reason = "I'm struggling with too many traps. Let me reduce them a bit."
            elif self.modifications['fruit_value'] < 15:
                modification = ('fruit_value', min(15, self.modifications['fruit_value'] + 2))
                reason = "I need more motivation. Let me increase the fruit value."
        
        # If doing well, make it more challenging
        elif avg_recent_score > 30 and improving:
            if self.modifications['trap_probability'] < 0.3:
                modification = ('trap_probability', min(0.3, self.modifications['trap_probability'] + 0.05))
                reason = "This is too easy. Let me add more traps for a challenge."
            elif self.modifications['max_steps'] < 300:
                modification = ('max_steps', min(300, self.modifications['max_steps'] + 20))
                reason = "I want more time to explore. Let me increase the max steps."
        
        # If learning has stagnated, adjust learning parameters
        elif not improving and len(older) > 0:
            if random.random() < 0.5:
                modification = ('learning_rate', max(0.0001, min(0.01, self.modifications['learning_rate'] * (0.8 if random.random() < 0.5 else 1.2))))
                reason = "My learning has stagnated. Let me adjust my learning rate."
            else:
                modification = ('epsilon_decay', max(0.99, min(0.999, self.modifications['epsilon_decay'] + (0.001 if random.random() < 0.5 else -0.001))))
                reason = "I need to change my exploration strategy."
        
        if modification:
            self.last_modification_time = time.time()
            self.modifications[modification[0]] = modification[1]
            self.modification_history.append({
                'parameter': modification[0],
                'value': modification[1],
                'reason': reason,
                'time': time.time()
            })
            
        return modification, reason
    
    def get_parameter(self, name):
        """Get the current value of a parameter"""
        return self.modifications.get(name, None)
    
    def save_state(self, filename='snake_self_awareness.json'):
        """Save the current state of modifications"""
        with open(filename, 'w') as f:
            json.dump({
                'modifications': self.modifications,
                'history': self.modification_history
            }, f, indent=2)
    
    def load_state(self, filename='snake_self_awareness.json'):
        """Load a previously saved state"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.modifications = data['modifications']
                self.modification_history = data['history']
            return True
        except:
            return False

# ---------------------------
# Environment: SnakeEnv
# ---------------------------
class SnakeEnv:
    def __init__(self, grid_width=20, grid_height=20, max_steps=200, render_mode=None, game_modifier=None):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.cell_size = 20  # pixels per cell
        self.game_modifier = game_modifier
        
        # Initialize text-to-speech engine
        self.tts_engine = None
        self.tts_thread = None
        self.last_spoken_thought = ""
        
        if render_mode == 'human':
            # Initialize text-to-speech engine with better error handling
            try:
                # Print system info for debugging
                print(f"Initializing TTS on {platform.system()} {platform.version()}")
                
                self.tts_engine = pyttsx3.init()
                # Test the engine
                self.tts_engine.say("Snake AI initialized")
                self.tts_engine.runAndWait()
                print("Text-to-speech initialized successfully")
                
                # Set properties
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 1.0)  # Maximum volume
                
                # Get available voices
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    print(f"Found {len(voices)} voices")
                    # Try to find a good voice
                    for voice in voices:
                        print(f"Available voice: {voice.name} ({voice.id})")
                    
                    # Try to set a voice
                    self.tts_engine.setProperty('voice', voices[0].id)
            except Exception as e:
                print(f"ERROR initializing text-to-speech: {e}")
                self.tts_engine = None
            
            pygame.init()
            # Increase window height to make room for dialogue
            self.screen = pygame.display.set_mode((grid_width * self.cell_size, 
                                                 grid_height * self.cell_size + 100))
            pygame.display.set_caption('Magic Snake AI')
            self.colors = {
                'background': (0, 0, 0),
                'snake': (0, 255, 0),
                'fruit': (255, 0, 0),
                'trap': (255, 255, 0),
                'text': (255, 255, 255),
                'dialogue': (255, 200, 100)
            }
            # Initialize font - make it larger and ensure it's initialized
            pygame.font.init()
            try:
                self.font = pygame.font.SysFont('Arial', 16)
                self.dialogue_font = pygame.font.SysFont('Arial', 18, bold=True)
            except:
                # Fallback to default font if Arial is not available
                self.font = pygame.font.Font(None, 16)
                self.dialogue_font = pygame.font.Font(None, 18)
        
        self.q_values = [0, 0, 0, 0]  # Store Q-values for visualization
        self.reset()

    def reset(self):
        # Update parameters from game_modifier if available
        if self.game_modifier:
            self.max_steps = self.game_modifier.get_parameter('max_steps') or self.max_steps
        
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

        # Calculate distance to fruit before moving
        head = self.snake[0]
        fruit_dist_before = abs(head[0] - self.fruit[0]) + abs(head[1] - self.fruit[1])
        
        new_head = (head[0] + move[0], head[1] + move[1])
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
            # Lose half the snake's length (rounded down).
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
            # Calculate distance to fruit after moving
            fruit_dist_after = abs(new_head[0] - self.fruit[0]) + abs(new_head[1] - self.fruit[1])
            # Add small reward for moving toward fruit
            if fruit_dist_after < fruit_dist_before:
                reward += 0.1  # Small positive reward for moving toward fruit

        # Add traps with probability from game_modifier if available
        trap_probability = 0.2  # default
        if self.game_modifier:
            trap_probability = self.game_modifier.get_parameter('trap_probability') or trap_probability
            
        if random.random() < trap_probability:
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

    def _get_q_interpretation(self, q_values):
        """Generate a varied and humorous interpretation of the snake's thinking"""
        directions = ["up", "right", "down", "left"]
        max_q = max(q_values)
        min_q = min(q_values)
        best_dir = directions[np.argmax(q_values)]
        worst_dir = directions[np.argmin(q_values)]
        
        # Calculate confidence level
        q_range = max(0.1, max_q - min_q)
        confidence = (max_q - np.mean(q_values)) / q_range
        
        # Varied and humorous messages based on different situations
        
        # High confidence messages
        high_confidence = [
            f"Definitely going {best_dir}! No doubt about it!",
            f"My snake senses tell me {best_dir} is the way to go!",
            f"{best_dir.upper()}! This is the way.",
            f"I'd bet my last mouse that {best_dir} is correct!",
            f"Going {best_dir} feels so right it should be illegal!",
            f"If I don't go {best_dir}, I'll eat my own tail!"
        ]
        
        # Medium confidence messages
        medium_confidence = [
            f"Hmm, I'm leaning towards {best_dir}, but don't quote me on that.",
            f"Eeny, meeny, miny, {best_dir}... I guess?",
            f"My gut says {best_dir}, but my gut has been wrong before.",
            f"When in doubt, go {best_dir}... and I'm definitely in doubt.",
            f"Is {best_dir} right? Let's find out! For science!",
            f"I'm 60% sure {best_dir} is correct. That's a passing grade, right?"
        ]
        
        # Confused messages
        confused_messages = [
            "I have no idea what I'm doing, but I'm doing it anyway!",
            "All directions look the same to me. Snake problems.",
            "Is this the real life? Is this just fantasy?",
            "I should have taken that left turn at Albuquerque...",
            "I'm not lost, I'm... exploring! Yeah, that's it.",
            "When did this game get so complicated?",
            "Brain.exe has stopped working. Send mice."
        ]
        
        # Avoiding bad direction messages
        avoiding_messages = [
            f"I'd rather eat broccoli than go {worst_dir}!",
            f"Going {worst_dir} seems like a terrible life choice.",
            f"My therapist told me to avoid going {worst_dir}.",
            f"Last time I went {worst_dir}, it didn't end well.",
            f"I have a phobia of going {worst_dir}. It's called common sense.",
            f"I sense great danger in the {worst_dir} direction."
        ]
        
        # Fruit-seeking messages (when fruit is nearby)
        fruit_messages = [
            "I smell fruit! Must... get... snack!",
            "Ooh, is that fruit I see? Come to mama!",
            "Fruit! The meaning of snake life!",
            "That fruit is calling my name. It's saying 'eat me'!",
            "Mmm, fruit. My favorite of the five food groups."
        ]
        
        # Trap-avoiding messages
        trap_messages = [
            "Traps everywhere! Who designed this game?!",
            "I'm surrounded by traps. This is fine.",
            "Navigating these traps is like snake Tetris!",
            "Traps to the left of me, traps to the right, here I am, stuck in the middle with food.",
            "I should have been a vegetarian. Less dangerous."
        ]
        
        # Check if fruit is nearby (high Q-value)
        if max_q > 5:
            return random.choice(fruit_messages)
        
        # Check if there are many traps (many negative Q-values)
        if sum(q < -2 for q in q_values) >= 2:
            return random.choice(trap_messages)
        
        # Select message based on confidence
        if confidence > 0.5:
            return random.choice(high_confidence)
        elif confidence > 0.2:
            return random.choice(medium_confidence)
        elif np.std(q_values) < 0.2:
            return random.choice(confused_messages)
        else:
            return random.choice(avoiding_messages)

    def _speak_thought(self, thought):
        """Speak the snake's thought using text-to-speech in a separate thread"""
        if self.tts_engine is None:
            return
            
        # Only speak if the thought is different
        if thought == self.last_spoken_thought:
            return
            
        print(f"Speaking: {thought}")  # Debug output
        
        # Stop any currently running speech
        if self.tts_thread and self.tts_thread.is_alive():
            # We can't directly stop pyttsx3, but we'll let the current speech finish
            self.tts_thread.join(0.1)  # Wait a bit but don't block
            
        # Force a direct synchronous speech for testing
        try:
            print("Attempting direct speech...")
            try:
                self.tts_engine.stop()  # Stop any previous speech
            except:
                pass  # Ignore if stop() fails
                
            self.tts_engine.say(thought)
            self.tts_engine.runAndWait()
            print("Direct speech completed")
        except Exception as e:
            print(f"Direct speech error: {e}")
            
            # Fall back to threaded speech
            def speak_text():
                try:
                    self.tts_engine.say(thought)
                    self.tts_engine.runAndWait()
                    print("Threaded speech completed")
                except Exception as e:
                    print(f"Threaded speech error: {e}")
            
            # Start speech in a new thread
            self.tts_thread = threading.Thread(target=speak_text)
            self.tts_thread.daemon = True
            self.tts_thread.start()
            
        self.last_spoken_thought = thought

    def render(self, q_values=None):
        if self.render_mode != 'human':
            return
            
        # Update Q-values if provided
        if q_values is not None:
            self.q_values = q_values
            
        # Handle Pygame events to prevent window from becoming unresponsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
        self.screen.fill(self.colors['background'])
        
        # Draw snake
        for i, segment in enumerate(self.snake):
            # Make head a different color (lighter green)
            color = (100, 255, 100) if i == 0 else self.colors['snake']
            pygame.draw.rect(self.screen, color,
                           (segment[0] * self.cell_size, 
                            segment[1] * self.cell_size,
                            self.cell_size, self.cell_size))
        
        # Draw fruit
        pygame.draw.rect(self.screen, self.colors['fruit'],
                        (self.fruit[0] * self.cell_size, 
                         self.fruit[1] * self.cell_size,
                         self.cell_size, self.cell_size))
        
        # Draw traps
        for trap in self.traps:
            pygame.draw.rect(self.screen, self.colors['trap'],
                           (trap[0] * self.cell_size, 
                            trap[1] * self.cell_size,
                            self.cell_size, self.cell_size))
        
        # Draw grid lines
        for x in range(0, self.grid_width * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, (50, 50, 50), (x, 0), (x, self.grid_height * self.cell_size))
        for y in range(0, self.grid_height * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, (50, 50, 50), (0, y), (self.grid_width * self.cell_size, y))
        
        # Draw info panel background
        info_panel_rect = pygame.Rect(0, self.grid_height * self.cell_size, 
                                     self.grid_width * self.cell_size, 100)
        pygame.draw.rect(self.screen, (30, 30, 30), info_panel_rect)
        
        # Draw Q-values at the bottom
        directions = ["Up", "Right", "Down", "Left"]
        for i, (direction, q_val) in enumerate(zip(directions, self.q_values)):
            # Highlight the highest Q-value
            color = (255, 255, 0) if q_val == max(self.q_values) else (255, 255, 255)
            text = self.font.render(f"{direction}: {q_val:.2f}", True, color)
            self.screen.blit(text, (10 + i * 100, self.grid_height * self.cell_size + 15))
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, self.grid_height * self.cell_size + 40))
        
        # Get snake's thought
        thought = self._get_q_interpretation(self.q_values)
        
        # Speak the thought if it's different from the last one
        self._speak_thought(thought)
        
        # Draw snake's "thoughts" as dialogue with a background
        thought_text = self.dialogue_font.render(f"Snake thinks: \"{thought}\"", True, self.colors['dialogue'])
        
        # Add a background for the dialogue text
        text_rect = thought_text.get_rect()
        text_rect.topleft = (10, self.grid_height * self.cell_size + 65)
        bg_rect = text_rect.inflate(20, 10)
        pygame.draw.rect(self.screen, (50, 50, 50), bg_rect, border_radius=5)
        
        # Draw the dialogue text
        self.screen.blit(thought_text, text_rect)
        
        pygame.display.flip()
        pygame.time.wait(100)  # Delay to make game visible

    def close(self):
        # Clean up text-to-speech when closing
        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(1.0)  # Give speech a chance to finish
            
        if self.render_mode == 'human':
            pygame.quit()

# ---------------------------
# DQN Agent
# ---------------------------
class DQNAgent:
    def __init__(self, state_shape, action_size, game_modifier=None):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.game_modifier = game_modifier
        
        # Update parameters from game_modifier if available
        if self.game_modifier:
            self.epsilon_decay = self.game_modifier.get_parameter('epsilon_decay') or self.epsilon_decay
            self.learning_rate = self.game_modifier.get_parameter('learning_rate') or self.learning_rate
            
        self.model = self._build_model()
        self.model_dir = 'saved_models'
        os.makedirs(self.model_dir, exist_ok=True)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential([
            # Reshape input to include channels dimension (1 channel)
            tf.keras.layers.Reshape((self.state_shape[0], self.state_shape[1], 1), input_shape=self.state_shape),
            
            # First Conv layer
            Conv2D(32, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            
            # Second Conv layer
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            
            # Third Conv layer
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            
            # Flatten and Dense layers
            Flatten(),
            Dense(256, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Add batch dimension for prediction
        state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state, verbose=0)
        
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            # For random actions, still return Q-values for visualization
            return action, act_values[0]
        
        return np.argmax(act_values[0]), act_values[0]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        # Predict Q-values for current states
        targets = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states
        next_q_values = self.model.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])

        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, episode=None):
        """Save the current model state"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_str = f"_episode_{episode}" if episode is not None else ""
        filename = f"snake_model_{timestamp}{episode_str}.h5"
        filepath = os.path.join(self.model_dir, filename)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        return filepath

    def load_model(self, filepath):
        """Load a previously saved model"""
        if os.path.exists(filepath):
            self.model = load_model(filepath)
            print(f"Model loaded from {filepath}")
            return True
        print(f"No model found at {filepath}")
        return False

# ---------------------------
# Helper Functions
# ---------------------------
def find_latest_model(model_dir='saved_models'):
    """Find the most recently saved model in the directory"""
    if not os.path.exists(model_dir):
        return None
        
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    if not model_files:
        return None
        
    # Sort by modification time (most recent first)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    return os.path.join(model_dir, model_files[0])

# ---------------------------
# Main Training Loop
# ---------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or run the Snake AI')
    parser.add_argument('--render', action='store_true', help='Enable visualization')
    parser.add_argument('--load-model', type=str, help='Path to saved model to load')
    parser.add_argument('--save-interval', type=int, default=100, 
                       help='Save model every N episodes')
    parser.add_argument('--auto-load', action='store_true', 
                       help='Automatically load the most recent model')
    parser.add_argument('--grid-size', type=int, default=30,
                       help='Size of the grid (width and height)')
    parser.add_argument('--cell-size', type=int, default=15,
                       help='Size of each cell in pixels')
    parser.add_argument('--test-audio', action='store_true',
                       help='Test audio system and exit')
    parser.add_argument('--self-aware', action='store_true',
                       help='Enable snake self-awareness (ability to modify game)')
    args = parser.parse_args()

    render_mode = 'human' if args.render else None
    
    # Use the grid size and cell size from command line arguments
    grid_width = args.grid_size
    grid_height = args.grid_size
    cell_size = args.cell_size
    
    # Initialize the game modifier if self-awareness is enabled
    game_modifier = None
    if args.self_aware:
        game_modifier = GameModifier()
        # Try to load previous state
        if not game_modifier.load_state():
            print("No previous self-awareness state found. Starting fresh.")
        else:
            print("Loaded previous self-awareness state.")
    
    env = SnakeEnv(grid_width=grid_width, grid_height=grid_height, 
                  max_steps=grid_width * grid_height,  # Scale max steps with grid size
                  render_mode=render_mode, game_modifier=game_modifier)
    
    # Set cell size if specified
    if render_mode == 'human':
        env.cell_size = cell_size
    
    initial_state = env.reset()
    state_shape = initial_state.shape
    action_size = 4
    agent = DQNAgent(state_shape, action_size, game_modifier=game_modifier)

    # Load model if specified
    if args.load_model:
        agent.load_model(args.load_model)
    # Auto-load most recent model if requested
    elif args.auto_load:
        latest_model = find_latest_model()
        if latest_model:
            print(f"Auto-loading most recent model: {latest_model}")
            agent.load_model(latest_model)
        else:
            print("No saved models found. Starting fresh training.")

    episodes = 2000  # Increased from 1000
    batch_size = 64  # Increased from 32

    # Test audio if requested
    if args.test_audio:
        try:
            print("Testing audio system...")
            engine = pyttsx3.init()
            engine.say("Snake audio test. If you can hear this, audio is working correctly.")
            engine.runAndWait()
            print("Audio test complete. Did you hear anything?")
        except Exception as e:
            print(f"Audio test failed: {e}")
        sys.exit(0)

    try:
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            for time in range(500):
                # Process Pygame events in the main loop too
                if args.render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            raise KeyboardInterrupt
                
                # Get action and Q-values
                action, q_values = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                
                if args.render:
                    env.render(q_values)
                    
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if done:
                    print("Episode: {}/{}, Score: {}, Total Reward: {:.1f}, Epsilon: {:.2f}"
                          .format(e, episodes, env.score, total_reward, agent.epsilon))
                    
                    # Log performance for self-awareness
                    if game_modifier:
                        game_modifier.log_performance(env.score, total_reward, time)
                        
                        # Check if the snake wants to modify the game
                        modification, reason = game_modifier.decide_modification()
                        if modification:
                            param_name, new_value = modification
                            print(f"\nSNAKE SELF-AWARENESS: Changing {param_name} to {new_value}")
                            print(f"Snake's reasoning: \"{reason}\"\n")
                            
                            # If the snake is modifying learning parameters, update the agent
                            if param_name in ['learning_rate', 'epsilon_decay']:
                                if param_name == 'learning_rate':
                                    # Need to recompile the model with new learning rate
                                    agent.learning_rate = new_value
                                    agent.model.compile(loss='mse', 
                                                      optimizer=Adam(learning_rate=new_value))
                                elif param_name == 'epsilon_decay':
                                    agent.epsilon_decay = new_value
                            
                            # Save the self-awareness state
                            game_modifier.save_state()
                    
                    break
                    
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
            # Save model at specified intervals
            if args.save_interval > 0 and (e + 1) % args.save_interval == 0:
                agent.save_model(episode=e + 1)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save final model and self-awareness state
        agent.save_model()
        if game_modifier:
            game_modifier.save_state()
    
    finally:
        env.close()
