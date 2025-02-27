 # Magic Snake - Self-Aware DQN Snake Game

   A reinforcement learning project where a snake learns to play the classic Snake game using Deep Q-Networks (DQN). The snake features self-awareness, allowing it to modify game parameters based on its performance.

   ## Features
   - Deep Q-Network agent with CNN layers
   - Self-modifying game parameters
   - Text-to-speech snake thoughts
   - Voice selection menu
   - Performance visualization

   ## Requirements
   See requirements.txt

   ## Usage
   
   ### Options
   - `--episodes N`: Number of episodes to train (default: 1000)
   - `--grid-size N`: Size of the grid (default: 20)
   - `--cell-size N`: Size of each cell in pixels (default: 20)
   - `--fps N`: Frames per second (default: 10)
   - `--load-model PATH`: Path to model to load
   - `--no-gui`: Disable GUI
   - `--no-speech`: Disable speech
   - `--select-voice`: Force voice selection

   ## Project Structure
   - `magic_snake.py`: Main script
   - `saved_models/`: Directory for saved models
   - `snake_self_awareness.json`: Self-awareness data