import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
import pygame
import random
import os
import datetime
from datetime import datetime as dt
import argparse
import threading
import pyttsx3
import json
import sys
import time
from collections import deque
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class GameModifier:
    """Class to allow the snake to modify game parameters based on its performance"""
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
        """Load the state from a file if it exists"""
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
        """Save the current state to a file"""
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
        """Log the performance of the snake"""
        timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
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
        """Consider changing parameters based on recent performance"""
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
        """Modify game parameters based on performance analysis"""
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
            'timestamp': dt.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        """Get the current parameters"""
        return self.current_parameters

def select_voice_interactive():
    """Interactive voice selection menu"""
    # Try to load existing preference
    voice_id = None
    if os.path.exists('snake_config.json'):
        try:
            with open('snake_config.json', 'r') as f:
                config = json.load(f)
                voice_id = config.get('voice_id')
                print(f"Found saved voice preference: {voice_id}")
        except Exception as e:
            print(f"Error loading voice config: {e}")
    
    # Check if we should force selection
    force_selection = '--select-voice' in sys.argv
    
    # If we have a saved preference and not forcing selection, return it
    if voice_id and not force_selection:
        try:
            # Verify the voice exists
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            voice_exists = any(v.id == voice_id for v in voices)
            if voice_exists:
                print(f"Using saved voice preference: {voice_id}")
                return voice_id
            else:
                print(f"Saved voice {voice_id} not found, selecting a new voice...")
        except Exception as e:
            print(f"Error checking saved voice: {e}")
    
    # Initialize TTS engine
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
    except Exception as e:
        print(f"Error initializing TTS engine: {e}")
        return None
    
    if not voices:
        print("No voices found in the TTS engine")
        return None
    
    # Categorize voices by gender
    male_voices = []
    female_voices = []
    other_voices = []
    
    for i, voice in enumerate(voices):
        voice_name = voice.name
        voice_id = voice.id
        
        # Try to determine gender
        if 'female' in voice_name.lower():
            female_voices.append((i, voice_name, voice_id))
        elif 'male' in voice_name.lower():
            male_voices.append((i, voice_name, voice_id))
        else:
            other_voices.append((i, voice_name, voice_id))
    
    # Display menu
    print("\n===== Voice Selection Menu =====")
    print("Male voices:")
    for i, name, _ in male_voices:
        print(f"{i}: {name}")
    
    print("\nFemale voices:")
    for i, name, _ in female_voices:
        print(f"{i}: {name}")
    
    print("\nOther voices:")
    for i, name, _ in other_voices:
        print(f"{i}: {name}")
    
    # Get user selection
    while True:
        try:
            selection = input("\nSelect voice by number (or press Enter for default): ")
            if not selection:
                # Default to first voice
                selected_voice = voices[0].id
                print(f"Using default voice: {voices[0].name}")
                break
            
            selection = int(selection)
            if 0 <= selection < len(voices):
                selected_voice = voices[selection].id
                print(f"Selected voice: {voices[selection].name}")
                
                # Test the voice
                print("Testing voice...")
                engine.setProperty('voice', selected_voice)
                engine.say("Hello, I am your snake assistant. Let's play a game!")
                engine.runAndWait()
                
                confirm = input("Use this voice? (y/n): ").lower()
                if confirm == 'y' or confirm == '':
                    break
            else:
                print("Invalid selection, please try again.")
        except ValueError:
            print("Please enter a number.")
        except Exception as e:
            print(f"Error during voice selection: {e}")
            return None
    
    # Save preference
    try:
        config = {'voice_id': selected_voice}
        with open('snake_config.json', 'w') as f:
            json.dump(config, f)
        print(f"Voice preference saved to snake_config.json")
    except Exception as e:
        print(f"Error saving voice preference: {e}")
    
    return selected_voice

class SnakeEnv:
    """Snake game environment for reinforcement learning"""
    
    def __init__(self, grid_size=20, cell_size=20, fps=10, use_gui=True, use_speech=True):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps
        self.use_gui = use_gui
        self.use_speech = use_speech
        
        # Initialize pygame first
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
        
        # Initialize TTS after pygame
        self.tts_engine = None
        self.tts_thread = None
        self.last_spoken_thought = None
        self.voice_id = None
        
        if self.use_speech:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 1.0)
                
                # Load voice preference if available
                if os.path.exists('snake_config.json'):
                    try:
                        with open('snake_config.json', 'r') as f:
                            config = json.load(f)
                            if 'voice_id' in config:
                                self.voice_id = config['voice_id']
                                self.tts_engine.setProperty('voice', self.voice_id)
                    except Exception as e:
                        print(f"Error loading voice config: {e}")
            except Exception as e:
                print(f"TTS initialization error: {e}")
                self.use_speech = False
        
        # Initialize game state variables
        self.snake = []
        self.direction = None
        self.fruit = None
        self.traps = []
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.thoughts = [
            "I should have taken that left turn at Albuquerque...",
            "When did this game get so complicated?",
            "I have no idea what I'm doing, but I'm doing it anyway!",
            "All directions look the same to me. Snake problems.",
            "I'm not lost, I'm... exploring! Yeah, that's it.",
            "Is that a fruit or a trap? Only one way to find out!",
            "Left, right, up, down... so many choices, so little time.",
            "If I keep going in circles, eventually I'll get somewhere... right?",
            "I'm starting to think this AI thing isn't all it's cracked up to be.",
            "Maybe I should have been a worm instead. Less pressure."
        ]
        
        # Game parameters
        self.trap_probability = 0.1
        self.reward_fruit = 1.0
        self.reward_trap = -1.0
        self.reward_collision = -1.0
        self.reward_step = -0.01
        
        # Now reset the game
        self.reset()
        
        # Test speech in a separate thread to avoid blocking
        if self.use_speech:
            def test_speech():
                try:
                    self._speak_thought(random.choice(self.thoughts))
                except Exception as e:
                    print(f"Initial speech test error: {e}")
            
            speech_test_thread = threading.Thread(target=test_speech)
            speech_test_thread.daemon = True
            speech_test_thread.start()
    
    def update_parameters(self, parameters):
        """Update game parameters"""
        self.trap_probability = parameters.get("trap_probability", self.trap_probability)
        self.reward_fruit = parameters.get("reward_fruit", self.reward_fruit)
        self.reward_trap = parameters.get("reward_trap", self.reward_trap)
        self.reward_collision = parameters.get("reward_collision", self.reward_collision)
        self.reward_step = parameters.get("reward_step", self.reward_step)
    
    def reset(self):
        """Reset the game state"""
        # Initialize snake in the middle of the grid
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        
        # Random initial direction
        self.direction = random.choice([(0, -1), (1, 0), (0, 1), (-1, 0)])  # Up, Right, Down, Left
        
        # Place fruit
        self.fruit = self._place_item()
        
        # Clear traps
        self.traps = []
        
        # Add initial traps
        for _ in range(int(self.grid_size * self.trap_probability)):
            trap = self._place_item()
            if trap:
                self.traps.append(trap)
        
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        return self._get_state()
    
    def _place_item(self):
        """Place an item (fruit or trap) on an empty cell"""
        empty_cells = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.snake and (x, y) != self.fruit and (x, y) not in self.traps:
                    empty_cells.append((x, y))
        
        if empty_cells:
            return random.choice(empty_cells)
        return None
    
    def _get_state(self):
        """Get the current state representation for the agent"""
        # Create a grid representation
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
        if 0 <= self.fruit[0] < self.grid_size and 0 <= self.fruit[1] < self.grid_size:
            state[self.fruit[1], self.fruit[0], 1] = 1
        
        # Traps
        for trap in self.traps:
            if 0 <= trap[0] < self.grid_size and 0 <= trap[1] < self.grid_size:
                state[trap[1], trap[0], 2] = 1
        
        return state
    
    def step(self, action):
        """Take a step in the environment based on the action"""
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
            self._speak_thought("Ouch! I ran into myself!")
        # Check for collision with trap
        elif new_head in self.traps:
            self.game_over = True
            reward = self.reward_trap
            self._speak_thought("Argh! That was a trap!")
        # Check for fruit
        elif new_head == self.fruit:
            self.snake.insert(0, new_head)
            self.score += 1
            reward = self.reward_fruit
            self._speak_thought("Yum! That was delicious!")
            
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
            
            # Occasionally speak a random thought
            if random.random() < 0.01:
                self._speak_thought(random.choice(self.thoughts))
        
        self.steps += 1
        
        # Check for game over due to too many steps
        if self.steps > 100 * self.grid_size and not self.game_over:
            self.game_over = True
            reward = self.reward_collision
            self._speak_thought("I'm getting tired... time for a nap.")
        
        return self._get_state(), reward, self.game_over, {"score": self.score}
    
    def _speak_thought(self, thought):
        """Speak the snake's thought using text-to-speech in a separate thread"""
        if not self.use_speech or self.tts_engine is None:
            return
            
        # Only speak if the thought is different
        if thought == self.last_spoken_thought:
            return
        
        # If we're already speaking, don't try to speak again
        if self.tts_thread and self.tts_thread.is_alive():
            return
            
        print(f"Speaking: {thought}")  # Debug output
        
        # Function to run in thread
        def speak_text():
            try:
                # Create a new engine instance for each speech to avoid "run loop already started" errors
                engine = pyttsx3.init()
                
                # Copy properties from the main engine
                if self.voice_id:
                    engine.setProperty('voice', self.voice_id)
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 1.0)
                
                engine.say(thought)
                engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")
        
        # Start speech in a new thread
        self.tts_thread = threading.Thread(target=speak_text)
        self.tts_thread.daemon = True
        self.tts_thread.start()
        self.last_spoken_thought = thought
    
    def render(self):
        """Render the game state"""
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
        """Clean up resources"""
        if self.tts_engine is not None:
            try:
                # Wait for any speech to finish
                if self.tts_thread and self.tts_thread.is_alive():
                    self.tts_thread.join(timeout=1.0)
            except Exception as e:
                print(f"Error closing TTS: {e}")
        
        # Only quit pygame if it was initialized
        if self.pygame_initialized:
            try:
                pygame.quit()
                self.pygame_initialized = False
            except Exception as e:
                print(f"Error closing pygame: {e}")

    @classmethod
    def create_safe(cls, grid_size=20, cell_size=20, fps=10, use_gui=True, use_speech=True):
        """Safely create an environment instance with fallbacks"""
        try:
            return cls(grid_size, cell_size, fps, use_gui, use_speech)
        except Exception as e:
            print(f"Error creating environment with GUI and speech: {e}")
            try:
                # Try without speech
                print("Trying without speech...")
                return cls(grid_size, cell_size, fps, use_gui, False)
            except Exception as e:
                print(f"Error creating environment with GUI: {e}")
                try:
                    # Try without GUI
                    print("Trying without GUI...")
                    return cls(grid_size, cell_size, fps, False, False)
                except Exception as e:
                    print(f"Fatal error creating environment: {e}")
                    raise

class DQNAgent:
    """Deep Q-Network agent for playing Snake"""
    
    def __init__(self, state_shape, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000, batch_size=32,
                 use_cnn=True):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_cnn = use_cnn
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """Build the neural network model"""
        if self.use_cnn:
            model = Sequential()
            model.add(Input(shape=self.state_shape))
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        else:
            # Flatten the input shape for a dense model
            flat_state_size = np.prod(self.state_shape)
            model = Sequential()
            model.add(Input(shape=self.state_shape))
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dense(256, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def update_target_model(self):
        """Update the target model to match the main model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, explore=True):
        """Choose an action based on the current state"""
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def get_q_values(self, state):
        """Get Q-values for visualization"""
        state = np.expand_dims(state, axis=0)
        return self.model.predict(state, verbose=0)[0]
    
    def replay(self, batch_size):
        """Train the model with experiences from memory"""
        try:
            if len(self.memory) < batch_size:
                return
            
            minibatch = random.sample(self.memory, batch_size)
            
            states = np.zeros((batch_size,) + self.state_shape)
            next_states = np.zeros((batch_size,) + self.state_shape)
            
            # Fill states and next_states
            for i, (state, _, _, next_state, _) in enumerate(minibatch):
                states[i] = state
                next_states[i] = next_state
            
            # Predict Q-values
            targets = self.model.predict(states, verbose=0)
            next_targets = self.target_model.predict(next_states, verbose=0)
            
            # Update targets with rewards
            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                if done:
                    targets[i, action] = reward
                else:
                    targets[i, action] = reward + self.gamma * np.max(next_targets[i])
            
            # Train the model
            self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        except Exception as e:
            print(f"Error during replay: {e}")

    def save_model(self, custom_name=None):
        """Save the model to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs('saved_models', exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            filename = custom_name if custom_name else f"snake_model_{timestamp}.h5"
            filepath = os.path.join('saved_models', filename)
            
            # Save the model
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving model: {e}")
            return None

    def load_model_safe(self, model_path):
        """Safely load a model with error handling"""
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
        
        try:
            self.model.load_weights(model_path)
            self.update_target_model()
            print(f"Successfully loaded model from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            
            # Try alternative loading method
            try:
                loaded_model = load_model(model_path)
                self.model.set_weights(loaded_model.get_weights())
                self.update_target_model()
                print(f"Successfully loaded model using alternative method")
                return True
            except Exception as e2:
                print(f"Alternative loading also failed: {e2}")
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
def main():
    """Main function to run the snake game with DQN agent"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Train a DQN agent to play Snake')
        parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
        parser.add_argument('--grid-size', type=int, default=20, help='Size of the grid')
        parser.add_argument('--cell-size', type=int, default=20, help='Size of each cell in pixels')
        parser.add_argument('--fps', type=int, default=10, help='Frames per second')
        parser.add_argument('--load-model', type=str, help='Path to model to load')
        parser.add_argument('--no-gui', action='store_true', help='Disable GUI')
        parser.add_argument('--no-speech', action='store_true', help='Disable speech')
        parser.add_argument('--select-voice', action='store_true', help='Force voice selection')
        args = parser.parse_args()
        
        # Initialize voice if speech is enabled
        use_speech = not args.no_speech
        if use_speech and (args.select_voice or not os.path.exists('snake_config.json')):
            voice_id = select_voice_interactive()
            # If voice selection failed and was required, disable speech
            if voice_id is None and args.select_voice:
                print("Voice selection failed. Disabling speech.")
                use_speech = False
        
        # Initialize environment
        grid_size = args.grid_size
        cell_size = args.cell_size
        env = SnakeEnv(grid_size=grid_size, cell_size=cell_size, fps=args.fps, 
                      use_gui=not args.no_gui, use_speech=use_speech)
        
        # Get state and action space info
        state = env.reset()
        state_shape = state.shape
        action_size = 4  # Up, Right, Down, Left
        
        # Initialize game modifier for self-awareness
        game_modifier = GameModifier()
        env.update_parameters(game_modifier.get_parameters())
        
        # Initialize agent
        agent = DQNAgent(state_shape, action_size)
        
        # Load model if specified
        if args.load_model and os.path.exists(args.load_model):
            try:
                agent.load_model_safe(args.load_model)
            except Exception as e:
                print(f"Error loading model: {e}")
        
        # Training loop
        episodes = args.episodes
        batch_size = 32
        update_target_every = 5
        
        total_rewards = []
        running = True
        
        # Set up logging
        def setup_logging(level=logging.INFO):
            """Set up logging configuration"""
            logging.basicConfig(
                level=level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler("snake_game.log"),
                    logging.StreamHandler()
                ]
            )
            return logging.getLogger('snake_game')

        logger = setup_logging()
        logger.info("Starting Snake Game with DQN Agent")
        
        try:
            for e in range(episodes):
                if not running:
                    break
                    
                state = env.reset()
                total_reward = 0
                
                while True:
                    # Process pygame events
                    if env.pygame_initialized:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                print("\nTraining interrupted by user (window closed)")
                                running = False
                                break
                    
                    if not running:
                        break
                    
                    # Get action
                    action = agent.act(state)
                    q_values = agent.get_q_values(state)  # Get q_values separately
                    
                    # Take action
                    next_state, reward, done, info = env.step(action)
                    
                    # Remember experience
                    agent.remember(state, action, reward, next_state, done)
                    
                    # Update state and reward
                    state = next_state
                    total_reward += reward
                    
                    # Render
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
                        game_modifier.log_performance(e, score, env.steps, total_reward)
                        
                        # Update environment parameters
                        env.update_parameters(game_modifier.get_parameters())
                        
                        total_rewards.append(total_reward)
                        break
                
                # Train the agent
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                
                # Decay epsilon
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
        
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
    
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

class ConfigManager:
    """Manage configuration settings for the game"""
    def __init__(self, config_file='snake_config.json'):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        default_config = {
            'voice_id': None,
            'grid_size': 20,
            'cell_size': 20,
            'fps': 10,
            'use_gui': True,
            'use_speech': True
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Update default with loaded values
                    default_config.update(loaded_config)
                    print(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                print(f"Error loading configuration: {e}")
        
        return default_config
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Saved configuration to {self.config_file}")
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def get(self, key, default=None):
        """Get a configuration value"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set a configuration value"""
        self.config[key] = value
        return self.save_config()
