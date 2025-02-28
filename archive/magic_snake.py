import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
import pygame
import random
import os
import datetime
from datetime import datetime as dt  # Import datetime class as dt to avoid confusion
import argparse
import threading
import pyttsx3
import json
import sys
import time
from collections import deque
import logging

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class GameModifier:
    """Class to allow the snake to modify game parameters based on its performance"""

    def __init__(self, file_path="snake_self_awareness.json", grid_size=30):
        self.file_path = file_path
        self.grid_size = grid_size
        self.performance_history = []
        self.parameter_changes = []

        # Scale parameters based on grid size
        trap_probability = 0.1 * (20 / grid_size)  # Adjust for larger grids

        self.current_parameters = {
            "trap_probability": trap_probability,
            "reward_fruit": 1.0,
            "reward_trap": -1.0,
            "reward_collision": -1.0,
            "reward_step": -0.01
            * (20 / grid_size),  # Smaller step penalty for larger grids
        }
        self.load_state()

    def load_state(self):
        """Load the state from the file"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    state = json.load(f)
                    self.current_parameters.update(state)
            except Exception as e:
                print(f"Error loading state: {e}")

    def save_state(self):
        """Save the state to the file"""
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.current_parameters, f)
        except Exception as e:
            print(f"Error saving state: {e}")

    def log_performance(self, episode, score, steps, total_reward):
        """Log performance data for analysis"""
        self.performance_history.append((episode, score, steps, total_reward))

    def get_parameters(self):
        """Get the current game parameters"""
        return self.current_parameters


def modify_parameters(self, episode, reason, score_change, reward_change):
    """Modify game parameters based on performance analysis"""
    old_params = self.current_parameters.copy()

    # Scale parameter changes based on grid size
    scale_factor = 20 / self.grid_size

    # If performance is declining, make it easier
    if score_change < 0:
        # Decrease trap probability
        self.current_parameters["trap_probability"] = max(
            0.05 * scale_factor,
            self.current_parameters["trap_probability"] - 0.02 * scale_factor,
        )
        # Increase fruit reward
        self.current_parameters["reward_fruit"] = min(
            1.5, self.current_parameters["reward_fruit"] + 0.1
        )
        # Decrease penalties
        self.current_parameters["reward_trap"] = min(
            -0.5, self.current_parameters["reward_trap"] + 0.1
        )
        self.current_parameters["reward_collision"] = min(
            -0.5, self.current_parameters["reward_collision"] + 0.1
        )
        self.current_parameters["reward_step"] = min(
            -0.005, self.current_parameters["reward_step"] + 0.005
        )

    # If performance is too good, make it harder
    elif score_change > 5:
        # Increase trap probability
        self.current_parameters["trap_probability"] = min(
            0.2, self.current_parameters["trap_probability"] + 0.02
        )
        # Decrease fruit reward
        self.current_parameters["reward_fruit"] = max(
            0.8, self.current_parameters["reward_fruit"] - 0.1
        )
        # Increase penalties
        self.current_parameters["reward_trap"] = max(
            -1.5, self.current_parameters["reward_trap"] - 0.1
        )
        self.current_parameters["reward_collision"] = max(
            -1.5, self.current_parameters["reward_collision"] - 0.1
        )
        self.current_parameters["reward_step"] = max(
            -0.02, self.current_parameters["reward_step"] - 0.005
        )

    # Random exploration of parameters occasionally
    elif random.random() < 0.3:
        param_to_change = random.choice(list(self.current_parameters.keys()))
        if param_to_change == "trap_probability":
            self.current_parameters[param_to_change] = max(
                0.05,
                min(
                    0.2,
                    self.current_parameters[param_to_change]
                    + random.uniform(-0.03, 0.03),
                ),
            )
        elif param_to_change in ["reward_fruit"]:
            self.current_parameters[param_to_change] = max(
                0.5,
                min(
                    1.5,
                    self.current_parameters[param_to_change]
                    + random.uniform(-0.2, 0.2),
                ),
            )
        elif param_to_change in ["reward_trap", "reward_collision"]:
            self.current_parameters[param_to_change] = max(
                -1.5,
                min(
                    -0.5,
                    self.current_parameters[param_to_change]
                    + random.uniform(-0.2, 0.2),
                ),
            )
        elif param_to_change == "reward_step":
            self.current_parameters[param_to_change] = max(
                -0.02,
                min(
                    -0.005,
                    self.current_parameters[param_to_change]
                    + random.uniform(-0.005, 0.005),
                ),
            )
        reason += f" (random exploration of {param_to_change})"

    # Log the change
    change = {
        "episode": episode,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason,
        "old_parameters": old_params,
        "new_parameters": self.current_parameters.copy(),
    }
    self.parameter_changes.append(change)

    print(f"\nSnake self-awareness update (Episode {episode}):")
    print(f"Reason: {reason}")
    print("Parameter changes:")
    for param in self.current_parameters:
        if old_params[param] != self.current_parameters[param]:
            print(
                f"  {param}: {old_params[param]:.4f} -> {self.current_parameters[param]:.4f}"
            )
    print()


# The DQNAgent.replay method was incomplete in our previous discussion
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


class SnakeEnv:
    """Snake game environment for reinforcement learning"""

    def __init__(self, grid_size=30, cell_size=15, fps=10, use_gui=True, use_speech=True):
        # Set basic attributes first to ensure they exist even if initialization fails
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps
        self.use_gui = use_gui
        self.use_speech = use_speech
        self.pygame_initialized = False
        self.tts_engine = None
        self.tts_thread = None
        self.last_spoken_thought = None
        self.voice_id = None
        
        # Initialize game state variables
        self.snake = []
        self.direction = None
        self.fruit = None
        self.traps = []
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        # Game parameters
        self.trap_probability = 0.1
        self.reward_fruit = 1.0
        self.reward_trap = -1.0
        self.reward_collision = -1.0
        self.reward_step = -0.01
        self.max_steps_factor = 5
        
        # Initialize thoughts
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
        
        # Initialize pygame
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
        
        # Now reset the game
        try:
            self.reset()
        except Exception as e:
            print(f"Error during initial reset: {e}")
        
        # Test speech in a separate thread to avoid blocking
        if self.use_speech and self.tts_engine:
            try:
                def test_speech():
                    try:
                        self._speak_thought(random.choice(self.thoughts))
                    except Exception as e:
                        print(f"Initial speech test error: {e}")
                
                speech_test_thread = threading.Thread(target=test_speech)
                speech_test_thread.daemon = True
                speech_test_thread.start()
            except Exception as e:
                print(f"Error starting speech test thread: {e}")

    def update_parameters(self, parameters):
        """Update game parameters"""
        # ... method code ...
        pass

    def reset(self):
        """Reset the game state"""
        try:
            # Verify grid_size exists and is valid
            if not hasattr(self, 'grid_size') or not isinstance(self.grid_size, int) or self.grid_size <= 0:
                print("Warning: Invalid grid_size, setting to default (30)")
                self.grid_size = 30
            
            # Initialize snake in the middle of the grid
            self.snake = [(self.grid_size // 2, self.grid_size // 2)]
            
            # Random initial direction
            self.direction = random.choice([(0, -1), (1, 0), (0, 1), (-1, 0)])  # Up, Right, Down, Left
            
            # Place fruit
            self.fruit = self._place_item()
            if self.fruit is None:
                # Fallback if no empty cell is found
                self.fruit = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
                while self.fruit in self.snake:
                    self.fruit = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            
            # Clear traps
            self.traps = []
            
            # Add initial traps
            if hasattr(self, 'trap_probability'):
                num_traps = int(self.grid_size * self.trap_probability)
                for _ in range(num_traps):
                    trap = self._place_item()
                    if trap:
                        self.traps.append(trap)
            
            self.score = 0
            self.steps = 0
            self.game_over = False
            
            # Get and return the state
            return self._get_state()
        
        except Exception as e:
            print(f"Error in reset: {e}")
            # Return a valid fallback state
            if hasattr(self, 'grid_size'):
                return np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
            else:
                print("Fatal error: grid_size not defined")
                return np.zeros((30, 30, 3), dtype=np.float32)  # Default fallback

    def _place_item(self):
        """Place an item (fruit or trap) on an empty cell"""
        # ... method code ...
        pass

    def _get_state(self):
        """Get the current state representation for the agent"""
        try:
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
            if self.fruit and 0 <= self.fruit[0] < self.grid_size and 0 <= self.fruit[1] < self.grid_size:
                state[self.fruit[1], self.fruit[0], 1] = 1
            
            # Traps
            for trap in self.traps:
                if 0 <= trap[0] < self.grid_size and 0 <= trap[1] < self.grid_size:
                    state[trap[1], trap[0], 2] = 1
            
            return state
        except Exception as e:
            print(f"Error in _get_state: {e}")
            # Return a valid fallback state
            return np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)

    def step(self, action):
        """Take a step in the environment based on the action"""
        # ... method code ...
        pass

    def _speak_thought(self, thought):
        """Speak the snake's thought using text-to-speech in a separate thread"""
        # ... method code ...
        pass

    def render(self):
        """Render the game state"""
        # ... method code ...
        pass

    def close(self):
        """Clean up resources"""
        # ... method code ...
        pass


def main():
    """Main function to run the snake game with DQN agent"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Train a DQN agent to play Snake")
        parser.add_argument(
            "--episodes", type=int, default=1000, help="Number of episodes to train"
        )
        parser.add_argument(
            "--grid-size", type=int, default=30, help="Size of the grid (default: 30)"
        )
        parser.add_argument(
            "--cell-size",
            type=int,
            default=15,
            help="Size of each cell in pixels (default: 15)",
        )
        parser.add_argument("--fps", type=int, default=10, help="Frames per second")
        parser.add_argument("--load-model", type=str, help="Path to model to load")
        parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
        parser.add_argument("--no-speech", action="store_true", help="Disable speech")
        parser.add_argument(
            "--select-voice", action="store_true", help="Force voice selection"
        )
        parser.add_argument(
            "--self-aware",
            action="store_true",
            help="Enable self-awareness (parameter adjustment)",
        )
        parser.add_argument(
            "--render",
            action="store_true",
            help="Force rendering even in training mode",
        )
        args = parser.parse_args()

        # Initialize environment with explicit error handling
        try:
            grid_size = args.grid_size
            cell_size = args.cell_size
            env = SnakeEnv(grid_size=grid_size, cell_size=cell_size, fps=args.fps, 
                          use_gui=not args.no_gui, use_speech=not args.no_speech)
            print(f"Environment initialized with grid_size={grid_size}")
        except Exception as e:
            print(f"Error initializing environment: {e}")
            return 1

        # Get state and action space info
        try:
            state = env.reset()
            if state is None:
                print("Error: Environment reset returned None state")
                return 1
            
            state_shape = state.shape
            action_size = 4  # Up, Right, Down, Left
            print(f"State shape: {state_shape}, Action size: {action_size}")
        except Exception as e:
            print(f"Error getting state shape: {e}")
            return 1

        # Initialize game modifier with grid size
        if args.self_aware:
            game_modifier = GameModifier(grid_size=args.grid_size)
            env.update_parameters(game_modifier.get_parameters())

        # Initialize agent with error handling
        try:
            print(f"Initializing DQN agent with state shape {state_shape}...")
            agent = DQNAgent(state_shape, action_size)
            print("DQN agent initialized successfully")
        except Exception as e:
            print(f"Error initializing agent: {e}")
            import traceback
            traceback.print_exc()
            return 1

        # Training loop
        episodes = args.episodes
        batch_size = 32
        update_target_every = 5
        
        total_rewards = []
        running = True
        
        print("Starting training loop...")
        
        try:
            for e in range(episodes):
                if not running:
                    break
                    
                print(f"Episode {e+1}/{episodes} starting...")
                
                try:
                    state = env.reset()
                    total_reward = 0
                    step_count = 0
                    
                    print(f"Environment reset, initial state shape: {state.shape}")
                    
        while True:
                        step_count += 1
                        
                        # Process pygame events
                        if env.pygame_initialized:
                            try:
                                for event in pygame.event.get():
                                    if event.type == pygame.QUIT:
                                        print("\nTraining interrupted by user (window closed)")
                                        running = False
                                        break
                            except Exception as e:
                                print(f"Error processing pygame events: {e}")
                        
                        if not running:
                            break
                        
                        # Get action
                        try:
                            action = agent.act(state)
                            # Debug print every 100 steps
                            if step_count % 100 == 0:
                                print(f"Step {step_count}, Action: {action}")
                        except Exception as e:
                            print(f"Error getting action: {e}")
                            running = False
                            break
                        
                        # Take action
                        try:
                            next_state, reward, done, info = env.step(action)
                        except Exception as e:
                            print(f"Error taking step: {e}")
                            running = False
                            break
                        
                        # Remember experience
                        try:
                            agent.remember(state, action, reward, next_state, done)
                        except Exception as e:
                            print(f"Error remembering experience: {e}")
                        
                        # Update state and reward
            state = next_state
            total_reward += reward
                        
                        # Render
                        try:
                            if args.render or not args.no_gui:
                                env.render()
                        except Exception as e:
                            print(f"Error rendering: {e}")
                        
            if done:
                            # Update target network periodically
                            if e % update_target_every == 0:
                                try:
                                    agent.update_target_model()
                                    print("Target model updated")
                                except Exception as e:
                                    print(f"Error updating target model: {e}")
                            
                            # Log performance
                            score = info.get('score', 0)
                            print(f"Episode: {e+1}/{episodes}, Score: {score}, " +
                                  f"Steps: {step_count}, Reward: {total_reward:.2f}, " +
                                  f"Epsilon: {agent.epsilon:.4f}")
                            
                            # Log to game modifier
                            if game_modifier:
                                try:
                                    game_modifier.log_performance(e, score, step_count, total_reward)
                                    env.update_parameters(game_modifier.get_parameters())
                                except Exception as e:
                                    print(f"Error updating game parameters: {e}")
                            
                            total_rewards.append(total_reward)
                break
                
                    # Train the agent
        if len(agent.memory) > batch_size:
                        try:
            agent.replay(batch_size)
                            if e % 10 == 0:  # Print every 10 episodes
                                print(f"Trained agent on batch of {batch_size} experiences")
                        except Exception as e:
                            print(f"Error during replay: {e}")
                
                    # Decay epsilon
                    if agent.epsilon > agent.epsilon_min:
                        agent.epsilon *= agent.epsilon_decay
                        
                except Exception as e:
                    print(f"Error during episode {e+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue to next episode
                    continue
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nTraining stopped due to error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("Training loop ended, cleaning up...")
            # Save final model
            try:
                print("Saving model...")
                agent.save_model()
                print("Model saved successfully")
            except Exception as e:
                print(f"Error saving model: {e}")
            
            if game_modifier:
                try:
                    print("Saving game modifier state...")
                    game_modifier.save_state()
                    print("Game modifier state saved successfully")
                except Exception as e:
                    print(f"Error saving game modifier state: {e}")
            
            try:
                print("Closing environment...")
                env.close()
                print("Environment closed successfully")
            except Exception as e:
                print(f"Error closing environment: {e}")
            
            # Plot training results if matplotlib is available
            try:
                if total_rewards:
                    print("Generating training rewards plot...")
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
        
            print("Program completed successfully")
            return 0

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
