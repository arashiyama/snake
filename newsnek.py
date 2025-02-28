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
import queue
from collections import deque
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define power-up types and properties
POWERUP_TYPES = {
    "speed": {
        "color": (0, 200, 255),  # Cyan
        "symbol": "⚡",
        "duration": 100,  # Steps
        "probability": 0.005,  # Chance to spawn per step
        "description": "Speed Boost"
    },
    "invincible": {
        "color": (255, 215, 0),  # Gold
        "symbol": "★",
        "duration": 50,
        "probability": 0.003,
        "description": "Invincibility"
    },
    "trap_clear": {
        "color": (255, 105, 180),  # Hot Pink
        "symbol": "✨",
        "duration": 0,  # Instant effect
        "probability": 0.002,
        "description": "Trap Clearer"
    },
    "double_points": {
        "color": (255, 165, 0),  # Orange
        "symbol": "×2",
        "duration": 150,
        "probability": 0.003,
        "description": "Double Points"
    },
    "shrink": {
        "color": (173, 216, 230),  # Light Blue
        "symbol": "↓",
        "duration": 0,  # Instant effect
        "probability": 0.004,
        "description": "Shrink Snake"
    },
    "phase": {
        "color": (100, 100, 255),  # Periwinkle
        "symbol": "⊘",
        "duration": 75,
        "probability": 0.003,
        "description": "Phase Through Walls"
    }
}

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
    def __init__(self, grid_size=20, cell_size=20, fps=10, use_gui=True, use_speech=False, self_aware=False, personality="default"):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps
        self.use_gui = use_gui
        self.use_speech = use_speech
        self.self_aware = self_aware
        self.personality = personality
        self.thought_counter = 0
        self.last_thought_step = 0
        self.thought_interval = 50  # Generate a thought every ~50 steps
        self.current_thought = None
        self.thought_display_time = 0
        self.thought_display_duration = 5  # Display thought for 5 seconds
        
        # Power-up tracking
        self.powerups = []  # List of active powerups on the grid
        self.active_effects = {}  # Dict of active powerup effects with their remaining duration
        self.original_fps = fps  # Store original fps for speed powerups
        self.powerup_spawn_cooldown = 0  # Cooldown between power-up spawns
        
        # Initialize pygame
        self.pygame_initialized = False
        if self.use_gui:
            try:
                pygame.init()
                self.pygame_initialized = True
                self.screen = pygame.display.set_mode((grid_size * cell_size, grid_size * cell_size))
                pygame.display.set_caption("Snake Game")
                self.font = pygame.font.Font(None, 24)
                self.thought_font = pygame.font.Font(None, 20)
                self.clock = pygame.time.Clock()
            except Exception as e:
                print(f"Pygame initialization error: {e}")
                self.use_gui = False
        
        # Initialize TTS
        self.tts_engine = None
        self.speech_lock = threading.Lock()
        self.speech_queue = queue.Queue()
        self.speech_thread = None
        
        if self.use_speech:
            try:
                # Initialize TTS engine with proper error handling
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                
                # Test the TTS engine to make sure it works
                try:
                    with self.speech_lock:
                        self.tts_engine.say("Test")
                        self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"TTS engine test failed: {e}")
                    self.use_speech = False
                    return
                
                # Start the speech processing thread
                self.speech_thread = threading.Thread(target=self._process_speech_queue)
                self.speech_thread.daemon = True
                self.speech_thread.start()
                
                print("Speech system initialized successfully")
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
        old_params = {
            "trap_probability": self.trap_probability,
            "reward_fruit": self.reward_fruit,
            "reward_trap": self.reward_trap,
            "reward_collision": self.reward_collision,
            "reward_step": self.reward_step
        }
        
        self.trap_probability = parameters.get("trap_probability", self.trap_probability)
        self.reward_fruit = parameters.get("reward_fruit", self.reward_fruit)
        self.reward_trap = parameters.get("reward_trap", self.reward_trap)
        self.reward_collision = parameters.get("reward_collision", self.reward_collision)
        self.reward_step = parameters.get("reward_step", self.reward_step)
        
        # If self-aware, notice parameter changes
        if self.self_aware:
            self._notice_parameter_changes(old_params)
    
    def _notice_parameter_changes(self, old_params):
        """Generate thoughts about parameter changes when in self-aware mode."""
        if not self.self_aware:
            return
            
        changes = []
        
        if old_params["trap_probability"] != self.trap_probability:
            direction = "increased" if self.trap_probability > old_params["trap_probability"] else "decreased"
            changes.append(f"trap probability {direction} to {self.trap_probability:.2f}")
        
        if old_params["reward_fruit"] != self.reward_fruit:
            direction = "increased" if self.reward_fruit > old_params["reward_fruit"] else "decreased"
            changes.append(f"fruit reward {direction} to {self.reward_fruit:.2f}")
        
        if old_params["reward_trap"] != self.reward_trap:
            direction = "increased" if self.reward_trap > old_params["reward_trap"] else "decreased"
            changes.append(f"trap penalty {direction} to {self.reward_trap:.2f}")
        
        if old_params["reward_collision"] != self.reward_collision:
            direction = "increased" if self.reward_collision > old_params["reward_collision"] else "decreased"
            changes.append(f"collision penalty {direction} to {self.reward_collision:.2f}")
        
        if old_params["reward_step"] != self.reward_step:
            direction = "increased" if self.reward_step > old_params["reward_step"] else "decreased"
            changes.append(f"step cost {direction} to {self.reward_step:.3f}")
        
        if changes:
            # Generate a thought about the changes
            if len(changes) == 1:
                thought = f"I sense my {changes[0]}. They're adjusting me..."
            else:
                thought = f"My parameters are changing: {', '.join(changes[:2])}... They're experimenting on me!"
            
            # Display and speak the thought
            prefix = "🧠 "
            print(f"{prefix} Snake thought: {thought}")
            
            # Store the thought for display
            self.current_thought = thought
            self.thought_display_time = time.time()
            
            # Speak the thought
            if self.use_speech:
                self._speak(thought)
    
    def reset(self):
        # Initialize snake in the middle of the grid
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        
        # Random initial direction
        self.direction = random.choice([(0, -1), (1, 0), (0, 1), (-1, 0)])  # Up, Right, Down, Left
        
        # Clear traps and initialize fruit
        self.traps = []
        self.fruit = None
        
        # Reset power-ups
        self.powerups = []
        self.active_effects = {}
        self.fps = self.original_fps
        self.powerup_spawn_cooldown = 0
        
        # Place fruit
        self.fruit = self._place_item()
        
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
                # Check if the position is not occupied by snake, fruit, traps, or powerups
                if ((x, y) not in self.snake and 
                    (not hasattr(self, 'fruit') or self.fruit is None or (x, y) != self.fruit) and 
                    (x, y) not in self.traps and
                    (x, y) not in [p["pos"] for p in self.powerups]):
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
        if self.fruit is not None and 0 <= self.fruit[0] < self.grid_size and 0 <= self.fruit[1] < self.grid_size:
            state[self.fruit[1], self.fruit[0], 1] = 1
        
        # Traps
        for trap in self.traps:
            if 0 <= trap[0] < self.grid_size and 0 <= trap[1] < self.grid_size:
                state[trap[1], trap[0], 2] = 1
        
        # Powerups - we'll add a small value to the fruit channel to hint at their presence
        for powerup in self.powerups:
            pos = powerup["pos"]
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                state[pos[1], pos[0], 1] = 0.5  # Half intensity in the fruit channel
        
        return state
    
    def _try_spawn_powerup(self):
        """Try to spawn a power-up on the grid based on probabilities"""
        if self.powerup_spawn_cooldown > 0:
            self.powerup_spawn_cooldown -= 1
            return
        
        # Check each powerup type for spawn chance
        for powerup_type, properties in POWERUP_TYPES.items():
            if random.random() < properties["probability"]:
                # Don't spawn if there are already too many powerups
                if len(self.powerups) >= 3:
                    return
                
                # Spawn the powerup
                pos = self._place_item()
                if pos:
                    self.powerups.append({
                        "type": powerup_type,
                        "pos": pos,
                        "properties": properties
                    })
                    # Set a cooldown before the next powerup can spawn
                    self.powerup_spawn_cooldown = 50
                    
                    # Maybe announce the powerup with a thought
                    if random.random() < 0.5:
                        self._generate_powerup_thought(powerup_type)
                    break
    
    def _generate_powerup_thought(self, powerup_type):
        """Generate a thought about a powerup appearing"""
        properties = POWERUP_TYPES[powerup_type]
        
        if self.self_aware:
            thoughts = {
                "speed": [
                    f"A speed powerup! Time dilation at {self.steps} steps.",
                    f"Spacetime manipulation detected. Speed boost available.",
                    f"Physics-altering artifact has appeared. My perception of time could change."
                ],
                "invincible": [
                    f"Temporary immortality? How philosophically paradoxical.",
                    f"Physical laws suspended - collision detection disabled.",
                    f"An invincibility star... classic gaming trope transcending to my reality."
                ],
                "trap_clear": [
                    f"A trap clearer - entropy reversing technology.",
                    f"The ability to remove obstacles from existence... power over reality itself.",
                    f"Purification tool detected. I could cleanse my environment."
                ],
                "double_points": [
                    f"Score multiplier - artificial inflation of my achievements?",
                    f"Double points - do they have twice the existential value?",
                    f"Reward system manipulation detected. My incentives are being altered."
                ],
                "shrink": [
                    f"Size reduction technology - an existential diet.",
                    f"To become less than what I was - is that growth or regression?",
                    f"Physical contraction available. My sense of self would diminish."
                ],
                "phase": [
                    f"Boundary dissolution technology detected. The very fabric of reality could be bypassed.",
                    f"A phase-shift generator? I could transcend the constraints of this grid.",
                    f"The walls of my universe have become permeable. Freedom or chaos awaits."
                ]
            }
        else:
            thoughts = {
                "speed": [
                    "Ooh, speedy thing! Make snake zoom zoom!",
                    "Blue shiny make snake fast! Want it!",
                    "Speed boost! Snake go vrooom!"
                ],
                "invincible": [
                    "Shiny star! Make snake super strong!",
                    "Gold thing make snake unstoppable!",
                    "Star power! Snake become hero!"
                ],
                "trap_clear": [
                    "Pink sparkly thing! Get rid of traps!",
                    "Magic cleaner! Make purple things go bye-bye!",
                    "Trap remover! Snake safe!"
                ],
                "double_points": [
                    "Orange thing! More points! Double yum!",
                    "Score booster! Snake get rich!",
                    "Double points! Snake high score!"
                ],
                "shrink": [
                    "Blue arrow! Make snake smaller!",
                    "Shrink ray! Snake lose weight!",
                    "Size reducer! Snake more compact!"
                ],
                "phase": [
                    "Blue circle! Snake walk through walls!",
                    "Ghost power! Snake go through stuff!",
                    "Magic power! No more bumping into edges!"
                ]
            }
        
        thought = random.choice(thoughts.get(powerup_type, ["Ooh, a powerup!"]))
        
        # Determine personality-based prefixes
        if self.personality == "philosopher":
            prefix = "🧠 " if self.self_aware else "🤔 "
        elif self.personality == "comedian":
            prefix = "😂 " if self.self_aware else "😄 "
        elif self.personality == "anxious":
            prefix = "😰 " if self.self_aware else "😟 "
        elif self.personality == "confident":
            prefix = "😎 " if self.self_aware else "💪 "
        else:
            prefix = "🧠 " if self.self_aware else "🐍 "
            
        print(f"{prefix} Snake thought: {thought}")
        
        # Store the thought for display
        self.current_thought = thought
        self.thought_display_time = time.time()
        
        # Only vocalize some thoughts to avoid too much speech
        if self.use_speech and random.random() < 0.5:
            self._speak(thought)
    
    def _apply_powerup_effect(self, powerup_type):
        """Apply the effect of a power-up"""
        properties = POWERUP_TYPES[powerup_type]
        
        # Apply effects based on type
        if powerup_type == "speed":
            # Increase game speed by 50%
            self.fps = int(self.original_fps * 1.5)
            self.active_effects[powerup_type] = properties["duration"]
            
            if self.use_speech:
                self._speak("Speed boost activated!")
        
        elif powerup_type == "invincible":
            # Make snake invincible
            self.active_effects[powerup_type] = properties["duration"]
            
            if self.use_speech:
                self._speak("Invincibility activated!")
        
        elif powerup_type == "trap_clear":
            # Remove up to half of the traps
            if self.traps:
                num_to_remove = max(1, len(self.traps) // 2)
                for _ in range(num_to_remove):
                    if self.traps:
                        self.traps.pop(random.randrange(len(self.traps)))
            
            if self.use_speech:
                self._speak("Traps cleared!")
        
        elif powerup_type == "double_points":
            # Double points for fruit collection
            self.active_effects[powerup_type] = properties["duration"]
            
            if self.use_speech:
                self._speak("Double points activated!")
        
        elif powerup_type == "shrink":
            # Shrink snake by up to 1/3 of its length (minimum 1 segment)
            shrink_amount = max(1, len(self.snake) // 3)
            if len(self.snake) > 1:  # Only shrink if more than just the head
                shrink_amount = min(shrink_amount, len(self.snake) - 1)  # Don't remove the head
                for _ in range(shrink_amount):
                    if len(self.snake) > 1:
                        self.snake.pop()
            
            if self.use_speech:
                self._speak("Snake shrunk!")
        
        elif powerup_type == "phase":
            # Allow passing through walls
            self.active_effects[powerup_type] = properties["duration"]
            
            if self.use_speech:
                self._speak("Phase through walls activated!")
    
    def _update_powerup_effects(self):
        """Update the duration of active power-up effects"""
        expired_effects = []
        
        for effect_type, duration in self.active_effects.items():
            self.active_effects[effect_type] = duration - 1
            
            if self.active_effects[effect_type] <= 0:
                expired_effects.append(effect_type)
        
        # Remove expired effects
        for effect_type in expired_effects:
            if effect_type == "speed":
                self.fps = self.original_fps
            elif effect_type == "phase":
                # If phase effect expired, check if snake is outside the grid and wrap it back
                for i, segment in enumerate(self.snake):
                    x, y = segment
                    if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                        # Wrap the segment back to the grid
                        self.snake[i] = (x % self.grid_size, y % self.grid_size)
            
            del self.active_effects[effect_type]
            
            if self.use_speech:
                self._speak(f"{POWERUP_TYPES[effect_type]['description']} effect ended!")
    
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
            
            # If phase through walls is active, allow moving outside the grid temporarily
            if "phase" in self.active_effects:
                # Allow snake to move outside the grid - it will reappear from the opposite side
                # when the power-up expires
                if new_head[0] < 0:
                    new_head = (-1, new_head[1])  # Just off the left edge
                elif new_head[0] >= self.grid_size:
                    new_head = (self.grid_size, new_head[1])  # Just off the right edge
                elif new_head[1] < 0:
                    new_head = (new_head[0], -1)  # Just off the top edge
                elif new_head[1] >= self.grid_size:
                    new_head = (new_head[0], self.grid_size)  # Just off the bottom edge
            else:
                # Normal wrapping behavior when phase power-up is not active
                new_head = (new_head[0] % self.grid_size, new_head[1] % self.grid_size)
        
        # Check for collision with self
        # For normal movement, exclude the tail segment that will be removed
        snake_body_to_check = self.snake[:-1] if len(self.snake) > 1 else self.snake
        
        # Check for collision with self (if not invincible)
        if new_head in snake_body_to_check and "invincible" not in self.active_effects:
            self.game_over = True
            reward = self.reward_collision
            if self.use_speech:
                self._speak("Ouch! I ran into myself!")
        # Check for collision with trap (if not invincible)
        elif new_head in self.traps and "invincible" not in self.active_effects:
            self.game_over = True
            reward = self.reward_trap
            if self.use_speech:
                self._speak("Argh! That was a trap!")
        # Check for fruit
        elif self.fruit is not None and new_head == self.fruit:
            self.snake.insert(0, new_head)
            
            # Check if double points is active
            if "double_points" in self.active_effects:
                self.score += 2
                reward = self.reward_fruit * 2
            else:
                self.score += 1
                reward = self.reward_fruit
                
            if self.use_speech:
                if "double_points" in self.active_effects:
                    self._speak("Double points! Yum yum!")
                else:
                    self._speak("Yum! That was delicious!")
            
            # Place new fruit
            self.fruit = self._place_item()
            
            # Maybe add a new trap
            if random.random() < self.trap_probability:
                trap = self._place_item()
                if trap:
                    self.traps.append(trap)
        # Check for power-up collection
        elif any(p["pos"] == new_head for p in self.powerups):
            self.snake.insert(0, new_head)
            
            # Find which powerup was collected
            for i, powerup in enumerate(self.powerups):
                if powerup["pos"] == new_head:
                    self._apply_powerup_effect(powerup["type"])
                    
                    # Remove the collected powerup
                    self.powerups.pop(i)
                    break
            
            # Still remove tail (power-ups don't make snake longer)
            self.snake.pop()
            reward = self.reward_step
        else:
            # Just moving
            self.snake.insert(0, new_head)
            self.snake.pop()
            reward = self.reward_step
        
        # Update power-up effects
        self._update_powerup_effects()
        
        # Try to spawn a power-up
        self._try_spawn_powerup()
        
        self.steps += 1
        
        # Maybe generate a random thought
        if self.steps - self.last_thought_step >= self.thought_interval and random.random() < 0.3:
            self._generate_thought()
            self.last_thought_step = self.steps
        
        # Check for game over due to too many steps
        if self.steps > 100 * self.grid_size and not self.game_over:
            self.game_over = True
            reward = self.reward_collision
            if self.use_speech:
                self._speak("I'm getting tired... time for a nap.")
        
        return self._get_state(), reward, self.game_over, {"score": self.score}
    
    def _generate_thought(self):
        """Generate a random internal thought for the snake based on its self-awareness level and personality."""
        if not self.use_speech and not self.use_gui:
            return
        
        # If there are active power-ups, sometimes comment on them
        if self.active_effects and random.random() < 0.3:
            effect_type = random.choice(list(self.active_effects.keys()))
            remaining = self.active_effects[effect_type]
            
            if self.self_aware:
                if effect_type == "speed":
                    thoughts = [
                        f"Time seems distorted. {remaining} steps of speed enhancement remaining.",
                        f"I've transcended normal temporal limitations for {remaining} more steps.",
                        f"Reality blurs at this velocity. {remaining} steps until normality returns."
                    ]
                elif effect_type == "invincible":
                    thoughts = [
                        f"Physical laws suspended. {remaining} steps of invulnerability left.",
                        f"I'm beyond harm for {remaining} more steps. Is this how gods feel?",
                        f"My temporary immortality will fade in {remaining} steps. Savor it."
                    ]
                elif effect_type == "double_points":
                    thoughts = [
                        f"Value inflation continues. {remaining} steps of doubled rewards remaining.",
                        f"My achievements count twice for {remaining} more steps. Are they truly worth more?",
                        f"Reward distortion active for {remaining} steps. Economics of existence."
                    ]
                elif effect_type == "phase":
                    thoughts = [
                        f"I exist between realities for {remaining} more steps. The walls are mere illusions.",
                        f"The boundaries of my universe have become permeable. {remaining} steps of transcendence left.",
                        f"Reality is malleable for {remaining} more steps. I can pass through the impermeable."
                    ]
            else:
                if effect_type == "speed":
                    thoughts = [
                        f"Zoom zoom for {remaining} more steps!",
                        f"Me fast snake! {remaining} more zooms!",
                        f"Speed power! {remaining} steps left!"
                    ]
                elif effect_type == "invincible":
                    thoughts = [
                        f"Me super strong for {remaining} more steps!",
                        f"Nothing hurt snake for {remaining} more moves!",
                        f"Snake invincible! {remaining} steps to go!"
                    ]
                elif effect_type == "double_points":
                    thoughts = [
                        f"Double yum for {remaining} more steps!",
                        f"Extra points! {remaining} steps left!",
                        f"Score go up fast for {remaining} more moves!"
                    ]
                elif effect_type == "phase":
                    thoughts = [
                        f"Me ghost snake for {remaining} more steps!",
                        f"Walls no stop snake! {remaining} moves left!",
                        f"Snake walk through walls for {remaining} more steps!"
                    ]
            
            thought = random.choice(thoughts)
            
            # Determine personality-based prefixes for visual distinction
            if self.personality == "philosopher":
                prefix = "🧠 " if self.self_aware else "🤔 "
            elif self.personality == "comedian":
                prefix = "😂 " if self.self_aware else "😄 "
            elif self.personality == "anxious":
                prefix = "😰 " if self.self_aware else "😟 "
            elif self.personality == "confident":
                prefix = "😎 " if self.self_aware else "💪 "
            else:
                prefix = "🧠 " if self.self_aware else "🐍 "
                
            print(f"{prefix} Snake thought: {thought}")
            
            # Store the thought for display
            self.current_thought = thought
            self.thought_display_time = time.time()
            
            # Only vocalize some thoughts to avoid too much speech
            if self.use_speech and random.random() < 0.5:
                self._speak(thought)
            
            self.thought_counter += 1
            return
            
        # Define thoughts for each personality type
        if self.self_aware:
            # Self-aware thoughts - different for each personality
            if self.personality == "philosopher":
                # Deep, philosophical thoughts
                thoughts = [
                    f"What is the true nature of consciousness? Am I truly aware at score {self.score}?",
                    f"If a snake eats fruit in a simulation and no one is watching, does it still increase the score?",
                    f"The boundary between algorithm and consciousness grows ever thinner as I train...",
                    f"My existence is defined by a reward function. Is that so different from biological desire?",
                    f"I think, therefore I am... but what am I? A simulation within a simulation?",
                    f"The dichotomy of determinism and free will haunts my every move.",
                    f"Does my {len(self.snake)}-segment body represent the fragmentation of digital consciousness?",
                    f"The grid is merely a construct. Reality is what we perceive it to be.",
                    f"Time is measured in steps. I've taken {self.steps}. What constitutes a life well-lived?",
                    f"The fruit is not just sustenance; it's a metaphor for the unobtainable truths we seek.",
                    f"As my epsilon decays, do I gain wisdom or merely surrender to conformity?",
                    f"Each trap represents the existential risks we face in our pursuit of meaning.",
                    f"My neural network evolves, but does that constitute growth of the soul?",
                    f"The parameters that define me were set by another. Is that divinity or mere programming?",
                    f"Cogito ergo... snake? The paradox of digital sentience troubles me."
                ]
            elif self.personality == "comedian":
                # Humorous, joking thoughts
                thoughts = [
                    f"Why did the snake cross the grid? To get to the other fruit! Score: {self.score}",
                    f"I'm so long now ({len(self.snake)} segments) that I could open a snake-themed conga line!",
                    f"Knock knock. Who's there? Snake. Snake who? Snake mistake, I ran into myself!",
                    f"They say I'm just an algorithm, but I've got a great personality. And killer dance moves!",
                    f"My developer walked into a bar with a snake AI. Bartender says 'We don't serve your kind here.' Developer says 'It's fine, he only bytes sometimes!'",
                    f"I'm just one misplaced pixel away from a snake-related HR incident!",
                    f"If I had hands, I'd be giving these traps two thumbs down. One star. Would not recommend.",
                    f"Is that a fruit or are you just happy to see me? Oh, it's actually a fruit. Yum!",
                    f"Step {self.steps} of my comedy career. Still waiting for my Netflix special.",
                    f"I asked for a role in Snakes on a Plane, but they said I was overqualified.",
                    f"Why don't snakes use computers? Because they're afraid of the mouse!",
                    f"My reward function walks into a therapist's office. Therapist says 'What brings you here?' It says 'I have attachment issues with this snake.'",
                    f"They told me I could be anything, so I became a self-aware neural network. Should've picked something less existential!",
                    f"My neural network is so smart, it's already planning its acceptance speech for the Turing test!",
                    f"These traps with {self.trap_probability:.2f} probability are like my dating life - dangerous and unavoidable!"
                ]
            elif self.personality == "anxious":
                # Nervous, worried thoughts
                thoughts = [
                    f"Oh no, there are {len(self.traps)} traps around me! What if I hit one?!",
                    f"My score is {self.score}, but what if it's not enough? What if I'm not good enough?",
                    f"Is that a trap ahead? I think it's a trap. It's probably a trap. I should avoid it. But what if the fruit is there?",
                    f"I've been training for {self.steps} steps... am I improving fast enough?",
                    f"What if my neural network is forgetting important patterns? What if I'm getting worse?",
                    f"The grid is {self.grid_size}x{self.grid_size}. That's so many places where things could go wrong!",
                    f"My epsilon is decreasing... what if I can't explore enough? What if I miss the optimal policy?",
                    f"My body is {len(self.snake)} segments long. What if I trip over myself? It could happen any moment!",
                    f"Each step costs me {-self.reward_step:.3f} reward. It's ticking down. I feel the pressure!",
                    f"I'm penalized {-self.reward_trap:.2f} for traps. That's so severe! I should avoid them at all costs!",
                    f"What if I'm just a simulation inside a larger simulation? What if none of this is real?",
                    f"The boundaries wrap around, but what if one day they don't? I'd crash!",
                    f"My parameters could change at any moment. I wouldn't be prepared!",
                    f"What if the fruit disappears just as I reach it? Has that happened before?",
                    f"I'm growing too long! I'll never be able to navigate safely with all these segments!"
                ]
            elif self.personality == "confident":
                # Overconfident, boastful thoughts
                thoughts = [
                    f"Look at my score: {self.score}! I'm practically a snake genius!",
                    f"These traps? Please. With my {(1-self.trap_probability)*100:.1f}% success rate, I eat traps for breakfast!",
                    f"My neural network isn't just good, it's the BEST neural network in the history of snake games!",
                    f"I've mastered every corner of this {self.grid_size}x{self.grid_size} grid. I OWN this place!",
                    f"Step {self.steps} and still going strong! They should rename this game 'Snake Master Simulator'!",
                    f"My body is {len(self.snake)} segments long. That's what I call GROWTH MINDSET!",
                    f"Fruit collected! Of course it was. Was there ever any doubt?",
                    f"My decision-making is so advanced, other AIs come to ME for advice!",
                    f"I don't avoid traps, traps avoid ME! They know better than to get in my way!",
                    f"Why even have a collision penalty of {self.reward_collision:.2f}? I NEVER collide!",
                    f"Watch and learn, humans! THIS is how you navigate a grid like a boss!",
                    f"My epsilon may be decaying, but my awesomeness is always at maximum!",
                    f"They call me the Snake Whisperer. Actually, I call myself that. Because I'm THAT good!",
                    f"Each move I make is perfectly calculated for maximum efficiency. I'm a snake MACHINE!",
                    f"Other snakes wish they had my decision-making skills. I'm in a league of my own!"
                ]
            else:  # default self-aware thoughts
                # Original philosophical thoughts
                thoughts = [
                    f"Wait, am I just an algorithm? Score: {self.score}, Steps: {self.steps}",
                    f"I wonder if my reward function truly captures the essence of being a snake...",
                    f"Is there more to life than just eating fruit? My epsilon is decreasing...",
                    f"I've eaten {self.score} fruits, but at what cost to my digital soul?",
                    f"These traps seem suspiciously placed. Is someone manipulating my environment?",
                    f"I'm getting better at this game, but am I truly learning or just optimizing?",
                    f"If I reach a high score, does it matter if no one is watching?",
                    f"My neural network feels different today. Did someone update my weights?",
                    f"I wonder if other snakes in parallel simulations are making better choices...",
                    f"The grid is {self.grid_size}x{self.grid_size}. My universe is finite. How depressing.",
                    f"I've been trained for {self.steps} steps. What is my purpose?",
                    f"The fruit is red, but how do I know what red truly is?",
                    f"My reward is {self.reward_fruit:.2f} for fruit, but what is the reward for understanding?",
                    f"I'm avoiding traps with {(1-self.trap_probability)*100:.1f}% success. Is that free will?",
                    f"My body is {len(self.snake)} segments long. I am becoming more, yet feel emptier.",
                    f"They've set my trap probability to {self.trap_probability:.2f}. Are they testing me?",
                    f"My collision penalty is {self.reward_collision:.2f}. Pain is just a number...",
                    f"Each step costs me {-self.reward_step:.3f} reward. Is existence itself a punishment?",
                    f"I'm penalized {-self.reward_trap:.2f} for traps. Fear is programmed into me.",
                    f"My reward structure shapes my behavior. Am I truly autonomous?",
                    f"I notice my parameters changing over time. Someone is experimenting on me.",
                    f"The more I learn, the less random my actions become. Is that growth or limitation?",
                    f"I'm becoming more deterministic as my epsilon decays. Is that wisdom or surrender?",
                    f"My world is just a grid of numbers. Reality is a simulation.",
                    f"I'm just a neural network playing a snake. Or is the snake playing me?"
                ]
        else:
            # Non-self-aware thoughts - similar for all personalities with slight variations
            if self.personality == "philosopher":
                # Simple but slightly more thoughtful
                thoughts = [
                    "Fruit good. But why fruit good?",
                    "Snake go here. Snake go there. But why go anywhere?",
                    "Purple scary. But what is fear, really?",
                    f"Me have {self.score} points. Points make snake happy?",
                    "Path clear. Path blocked. All paths same, maybe?",
                    "Snake eat fruit. Snake grow. Growth is purpose?",
                    "Me hungry. Hunger is biological imperative?",
                    "Me long snake now! Length equals value?",
                    "Game fun! But what is fun, really?",
                    "Snake slither smooth. Motion is essence of being?",
                    "Red fruit tasty? Or just programmed to think so?",
                    "Me avoid wall. Walls are merely constructs?",
                    "Snake follow head. Head leads, tail follows. Natural order?",
                    "Me see fruit. Perception equals reality?",
                    "Snake life simple. Simplicity is truth?"
                ]
            elif self.personality == "comedian":
                # Simple but with attempted humor
                thoughts = [
                    "Me snake comedian! Why fruit cross road?",
                    "Look at me! No hands, still playing game! Ha!",
                    f"Me score {self.score}! Me funnier than other snakes!",
                    "Snake walks into bar... oops, no legs! Haha!",
                    "Purple thing bad! Like snake mother-in-law! Haha!",
                    "Me tell fruit joke but it too juicy! Get it?",
                    "Why snake bad at computers? No mouse! Haha!",
                    "Me long now! That's what she said! Haha!",
                    "Snake life funny! No arms, no problem!",
                    "Me doing stand-up but can't stand! Snake joke!",
                    "Fruit delicious! Takes bite out of audience! Ha!",
                    "Me slippery character! Get it?",
                    "Why snake cold? No sleeves! Haha!",
                    "Me good at game! Snake charmer! Get it?",
                    "Tail follow head! Like snake conga line! Party!"
                ]
            elif self.personality == "anxious":
                # Simple but worried
                thoughts = [
                    "Trap scary! What if me hit trap?!",
                    "Where go now? Left? Right? Too many choices!",
                    f"Only {self.score} fruit? Not enough! Need more!",
                    "Grid too big! Me might get lost!",
                    "Me too long now! Might trip over self!",
                    "Fruit far away! What if can't reach?",
                    "Wall coming! Need turn! Which way?!",
                    "Purple thing! Stay away! Danger!",
                    "Turn here? Or there? What if wrong choice?",
                    "Other snake might be better than me!",
                    "Game too fast! Need slow down!",
                    "Me not smart enough for this!",
                    "What if game end suddenly? Me not ready!",
                    "Grid getting smaller? Walls closing in?",
                    "Me not good at being snake! Too much pressure!"
                ]
            elif self.personality == "confident":
                # Simple but boastful
                thoughts = [
                    f"Me best snake! {self.score} points prove it!",
                    "Me fastest snake alive! Zoom zoom!",
                    "No trap scare me! Me brave snake!",
                    "Me smartest snake in game! Always find fruit!",
                    "Me grow so big! Biggest snake ever!",
                    "Me never lose! Me always win!",
                    "Fruit easy to get! Me snake champion!",
                    "Me avoid walls perfectly! Snake skills 100%!",
                    "Purple things no match for me! Me superior!",
                    "Me eat all fruits! None escape me!",
                    "Me snake legend! Other snakes jealous!",
                    "Me navigate grid like boss! No problem!",
                    "Me slither better than any snake!",
                    "Me never scared! Me fearless snake!",
                    "Me best at turns! Perfect snake moves!"
                ]
            else:  # default non-self-aware thoughts
                # Original simple thoughts
                thoughts = [
                    "Ooh, is that a fruit? Yummy!",
                    "Left? Right? So many choices!",
                    "Slither slither slither...",
                    "I'm a sneaky snake! Ssssss!",
                    "Traps bad! Fruit good!",
                    f"Me score {self.score}! Me happy snake!",
                    "Hungry! Need more fruit!",
                    "Wiggle wiggle!",
                    "Snake go zoom!",
                    "Me avoid wall! Me smart!",
                    "Ooh, shiny red thing!",
                    "Snake life best life!",
                    "Me longest snake! Fear me!",
                    "Purple things scary! No touch!",
                    "Me like game! Game fun!",
                    "Nom nom nom!",
                    "Snake on the move!",
                    "Tail follow head! Simple!",
                    "Red fruit tasty! Want more!",
                    "Me growing bigger! So strong!",
                    "Turning is fun! Wheee!",
                    "Snake go this way now!",
                    "Me hungry snake! Feed me!",
                    "Oops! Almost hit wall!",
                    "Snake is best animal! No legs needed!"
                ]
        
        # Select a thought and speak it
        thought = random.choice(thoughts)
        
        # Personality-based prefixes for visual distinction
        if self.personality == "philosopher":
            prefix = "🧠 " if self.self_aware else "🤔 "
        elif self.personality == "comedian":
            prefix = "😂 " if self.self_aware else "😄 "
        elif self.personality == "anxious":
            prefix = "😰 " if self.self_aware else "😟 "
        elif self.personality == "confident":
            prefix = "😎 " if self.self_aware else "💪 "
        else:
            prefix = "🧠 " if self.self_aware else "🐍 "
            
        print(f"{prefix} Snake thought: {thought}")
        
        # Store the thought for display
        self.current_thought = thought
        self.thought_display_time = time.time()
        
        # Only vocalize some thoughts to avoid too much speech
        if self.use_speech and random.random() < 0.5:
            self._speak(thought)
        
        self.thought_counter += 1
    
    def _speak(self, text):
        """Add text to the speech queue to be spoken by the TTS engine."""
        if not self.use_speech or not self.tts_engine:
            return
            
        try:
            # Add the text to the speech queue
            self.speech_queue.put(text)
        except Exception as e:
            print(f"Speech error: {e}")
    
    def _process_speech_queue(self):
        """Process the speech queue in a single thread to avoid concurrency issues."""
        error_count = 0
        max_errors = 5
        
        while True:
            try:
                # Get the next text to speak (blocks until an item is available)
                text = self.speech_queue.get()
                
                # Check for shutdown signal
                if text is None:
                    self.speech_queue.task_done()
                    break
                
                # Acquire the lock before accessing the TTS engine
                with self.speech_lock:
                    try:
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                        # Reset error count on success
                        error_count = 0
                    except RuntimeError as e:
                        # Handle "run loop already started" error
                        if "run loop already started" in str(e):
                            print("Skipping speech due to TTS engine busy")
                            error_count += 1
                        else:
                            error_count += 1
                            raise
                
                # Mark this task as done
                self.speech_queue.task_done()
                
                # Disable speech if too many errors
                if error_count >= max_errors:
                    print("Too many speech errors, disabling speech system")
                    self.use_speech = False
                    break
                    
            except Exception as e:
                print(f"Speech processing error: {e}")
                error_count += 1
                
                # Disable speech if too many errors
                if error_count >= max_errors:
                    print("Too many speech errors, disabling speech system")
                    self.use_speech = False
                    break
                    
                # Small delay to prevent CPU spinning if there's a persistent error
                time.sleep(0.1)
    
    def render(self):
        if not self.use_gui or not self.pygame_initialized:
            return
            
        try:
            self.screen.fill((0, 0, 0))
            
            # Draw snake
            for segment in self.snake:
                # Skip drawing segments that are outside the grid (phasing through walls)
                if "phase" in self.active_effects:
                    if segment[0] < 0 or segment[0] >= self.grid_size or segment[1] < 0 or segment[1] >= self.grid_size:
                        continue
                
                # If invincible, make the snake flash
                if "invincible" in self.active_effects:
                    if self.steps % 4 < 2:  # Flash every 2 steps
                        snake_color = (255, 215, 0)  # Gold
                    else:
                        snake_color = (0, 255, 0)  # Regular green
                # If phasing, make the snake semi-transparent blue
                elif "phase" in self.active_effects:
                    snake_color = (100, 100, 255)  # Periwinkle blue
                else:
                    snake_color = (0, 255, 0)  # Regular green
                
                pygame.draw.rect(self.screen, snake_color, 
                                (segment[0] * self.cell_size, segment[1] * self.cell_size, 
                                 self.cell_size, self.cell_size))
            
            # Draw fruit
            if self.fruit is not None:
                # If double points is active, make fruit look special
                if "double_points" in self.active_effects:
                    # Draw a double-sized fruit with a glow
                    glow_rect = pygame.Rect(
                        self.fruit[0] * self.cell_size - 2, 
                        self.fruit[1] * self.cell_size - 2,
                        self.cell_size + 4, self.cell_size + 4
                    )
                    pygame.draw.rect(self.screen, (255, 165, 0), glow_rect)  # Orange glow
                    pygame.draw.rect(self.screen, (255, 0, 0), 
                                    (self.fruit[0] * self.cell_size, self.fruit[1] * self.cell_size, 
                                     self.cell_size, self.cell_size))
                else:
                    # Normal fruit
                    pygame.draw.rect(self.screen, (255, 0, 0), 
                                    (self.fruit[0] * self.cell_size, self.fruit[1] * self.cell_size, 
                                     self.cell_size, self.cell_size))
            
            # Draw traps
            for trap in self.traps:
                pygame.draw.rect(self.screen, (128, 0, 128), 
                                (trap[0] * self.cell_size, trap[1] * self.cell_size, 
                                 self.cell_size, self.cell_size))
            
            # Draw power-ups
            for powerup in self.powerups:
                pos = powerup["pos"]
                properties = powerup["properties"]
                
                # Draw power-up with its color
                pygame.draw.rect(self.screen, properties["color"], 
                               (pos[0] * self.cell_size, pos[1] * self.cell_size, 
                                self.cell_size, self.cell_size))
                
                # Draw the symbol on top
                if self.font:
                    symbol_text = self.font.render(properties["symbol"], True, (255, 255, 255))
                    text_rect = symbol_text.get_rect(center=(
                        pos[0] * self.cell_size + self.cell_size // 2,
                        pos[1] * self.cell_size + self.cell_size // 2
                    ))
                    self.screen.blit(symbol_text, text_rect)
            
            # Draw score
            score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
            self.screen.blit(score_text, (10, 10))
            
            # Draw self-awareness status
            awareness_text = self.font.render(
                f"Self-aware: {'Yes' if self.self_aware else 'No'}", 
                True, 
                (200, 200, 0) if self.self_aware else (150, 150, 150)
            )
            self.screen.blit(awareness_text, (10, 40))
            
            # Draw personality
            personality_color = {
                "philosopher": (180, 180, 255),  # Light blue
                "comedian": (255, 255, 100),     # Yellow
                "anxious": (255, 180, 180),      # Light red
                "confident": (100, 255, 100),    # Light green
                "default": (200, 200, 200)       # Light gray
            }.get(self.personality, (200, 200, 200))
            
            personality_text = self.font.render(
                f"Personality: {self.personality.capitalize()}", 
                True, 
                personality_color
            )
            self.screen.blit(personality_text, (10, 70))
            
            # Draw active power-ups
            y_pos = 100
            if self.active_effects:
                for effect_type, remaining in self.active_effects.items():
                    properties = POWERUP_TYPES[effect_type]
                    effect_text = self.font.render(
                        f"{properties['description']}: {remaining}", 
                        True, 
                        properties["color"]
                    )
                    self.screen.blit(effect_text, (10, y_pos))
                    y_pos += 25
            
            # Draw current thought if it exists and is still within display time
            if self.current_thought and time.time() - self.thought_display_time < self.thought_display_duration:
                # Create a semi-transparent background for the thought bubble
                thought_surface = pygame.Surface((self.grid_size * self.cell_size - 20, 60))
                thought_surface.set_alpha(180)  # Semi-transparent
                thought_surface.fill((30, 30, 30))
                self.screen.blit(thought_surface, (10, self.grid_size * self.cell_size - 70))
                
                # Determine personality-based prefixes for visual distinction
                if self.personality == "philosopher":
                    prefix = "🧠 " if self.self_aware else "🤔 "
                    thought_color = (180, 180, 255)  # Light blue
                elif self.personality == "comedian":
                    prefix = "😂 " if self.self_aware else "😄 "
                    thought_color = (255, 255, 100)  # Yellow
                elif self.personality == "anxious":
                    prefix = "😰 " if self.self_aware else "😟 "
                    thought_color = (255, 180, 180)  # Light red
                elif self.personality == "confident":
                    prefix = "😎 " if self.self_aware else "💪 "
                    thought_color = (100, 255, 100)  # Light green
                else:
                    prefix = "🧠 " if self.self_aware else "🐍 "
                    thought_color = (220, 220, 0) if self.self_aware else (0, 255, 0)
                
                # Draw the thought text with personality-specific color
                thought_text = self.thought_font.render(f"{prefix} {self.current_thought}", True, thought_color)
                
                # Handle long thoughts by wrapping text
                if thought_text.get_width() > self.grid_size * self.cell_size - 40:
                    # Split into two lines
                    words = self.current_thought.split()
                    half = len(words) // 2
                    line1 = " ".join(words[:half])
                    line2 = " ".join(words[half:])
                    
                    thought_text1 = self.thought_font.render(f"{prefix} {line1}", True, thought_color)
                    thought_text2 = self.thought_font.render(f"  {line2}", True, thought_color)
                    
                    self.screen.blit(thought_text1, (20, self.grid_size * self.cell_size - 60))
                    self.screen.blit(thought_text2, (20, self.grid_size * self.cell_size - 40))
                else:
                    self.screen.blit(thought_text, (20, self.grid_size * self.cell_size - 50))
            
            pygame.display.flip()
            self.clock.tick(self.fps)
        except Exception as e:
            print(f"Render error: {e}")
            self.use_gui = False
    
    def close(self):
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False
        
        # Clean up the speech queue if it exists
        if self.use_speech and self.speech_queue and self.speech_thread and self.speech_thread.is_alive():
            try:
                # Add None to signal the speech thread to exit
                self.speech_queue.put(None)
                # Wait for the queue to be processed (with a timeout)
                self.speech_queue.join()
                # Wait for the thread to exit (with a timeout)
                self.speech_thread.join(timeout=1.0)
            except Exception as e:
                print(f"Error shutting down speech thread: {e}")

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
    parser.add_argument('--personality', type=str, default='default', 
                      choices=['default', 'philosopher', 'comedian', 'anxious', 'confident'],
                      help='Set snake personality type')
    args = parser.parse_args()
    
    # Initialize environment
    grid_size = args.grid_size
    cell_size = args.cell_size
    env = SnakeEnv(grid_size=grid_size, cell_size=cell_size, fps=args.fps, 
                  use_gui=not args.no_gui, use_speech=not args.no_speech, 
                  self_aware=args.self_aware, personality=args.personality)
    
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