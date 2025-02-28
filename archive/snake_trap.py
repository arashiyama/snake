import pygame
import random
import heapq
import sys

# -------- Configuration --------
CELL_SIZE    = 20      # Pixel size of one grid cell
GRID_WIDTH   = 20      # Number of cells horizontally
GRID_HEIGHT  = 20      # Number of cells vertically
WINDOW_WIDTH  = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 10              # Game speed (frames per second)
PENALTY_SEGMENTS = 3  # Number of segments to remove on trap collision

# Colors (R, G, B)
BLACK   = (0, 0, 0)
GREEN   = (0, 255, 0)
RED     = (255, 0, 0)
BLUE    = (0, 0, 255)
YELLOW  = (255, 255, 0)
ORANGE  = (255, 165, 0)  # Color for traps
WHITE   = (255, 255, 255)

# -------- Fruit Generation --------
def get_random_fruit(snake):
    """
    Returns a dictionary representing a fruit with a random type,
    color, and point value that is not placed on the snake.
    
    Fruit types:
      - Red fruit:    10 points
      - Blue fruit:   20 points
      - Yellow fruit: 5 points
    """
    fruit_types = [
        {"color": RED,    "value": 10},
        {"color": BLUE,   "value": 20},
        {"color": YELLOW, "value": 5}
    ]
    fruit_type = random.choice(fruit_types)
    
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in snake:
            break

    return {"pos": pos, "color": fruit_type["color"], "value": fruit_type["value"]}

def get_random_trap(snake, fruit, traps):
    """
    Returns a random grid coordinate for a trap that is not on the snake,
    the current fruit, or an existing trap.
    """
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in snake and pos != fruit["pos"] and pos not in traps:
            return pos

# -------- Utility Functions --------
def heuristic(a, b):
    """Manhattan distance heuristic for A*."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos):
    """Return the list of neighboring positions (up, down, left, right) for a given cell."""
    x, y = pos
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def a_star(start, goal, snake):
    """
    Returns a list of grid positions (a path) from start to goal using A* path-finding.
    The snake's body (except for the tail) is considered an obstacle.
    """
    obstacles = set(snake[:-1]) if len(snake) > 1 else set()
    
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    
    came_from = {}
    g_score = {start: 0}
    closed_set = set()
    
    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path from start to goal
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        closed_set.add(current)
        
        for neighbor in get_neighbors(current):
            x, y = neighbor
            if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
                continue
            if neighbor in obstacles and neighbor != goal:
                continue
                
            tentative_g = current_g + 1
            if neighbor in closed_set and tentative_g >= g_score.get(neighbor, float('inf')):
                continue
                
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                
    return None  # No valid path found

def is_move_safe(new_head, snake, fruit):
    """
    Simulate the snake’s move and check whether the new head position will
    allow the snake to eventually reach its tail (i.e. not trap itself).
    The simulation assumes that if the snake is not eating fruit,
    its tail will be removed.
    """
    # Simulate new snake state:
    if new_head == fruit["pos"]:
        # Eating fruit: snake grows (tail remains)
        new_snake = [new_head] + snake[:]
    else:
        # Normal move: snake moves (tail is removed)
        new_snake = [new_head] + snake[:-1]
    
    # For connectivity, treat all new_snake segments except the tail as obstacles.
    obstacles = set(new_snake[:-1])
    target = new_snake[-1]  # The tail cell we want to reach

    # Use Breadth-First Search (BFS) from new_head to target.
    queue = [new_head]
    visited = set([new_head])
    while queue:
        current = queue.pop(0)
        if current == target:
            return True
        for neighbor in get_neighbors(current):
            x, y = neighbor
            if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
                continue
            if neighbor == target:
                return True
            if neighbor in obstacles:
                continue
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False

def get_safe_move(snake, fruit):
    """
    Fallback strategy: from the snake's head, try all four directions and return
    the first move that is both immediately safe (no wall or self collision)
    and passes the connectivity (tail-reaching) check.
    
    If no candidate passes the connectivity test, return any immediate safe move.
    """
    head = snake[0]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    immediate_safe_moves = []
    for dx, dy in directions:
        new_head = (head[0] + dx, head[1] + dy)
        # Check boundaries
        if new_head[0] < 0 or new_head[0] >= GRID_WIDTH or new_head[1] < 0 or new_head[1] >= GRID_HEIGHT:
            continue
        # Check immediate collision with self
        if new_head in snake:
            continue
        immediate_safe_moves.append(new_head)
        if is_move_safe(new_head, snake, fruit):
            return new_head
    # If none passes the connectivity test, choose any immediate safe move.
    if immediate_safe_moves:
        return immediate_safe_moves[0]
    return head  # No safe move found; this will result in collision.

# -------- Main Game Loop --------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Self-Playing Snake Game with Traps")
    clock = pygame.time.Clock()
    
    # Set up a font for displaying the score.
    font = pygame.font.SysFont("arial", 20)
    
    # Initialize the snake in the center of the grid.
    snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
    
    # Generate the first fruit.
    fruit = get_random_fruit(snake)
    
    score = 0  # Initialize score
    
    # Initialize trap storage and timer event (trap appears every 2000 ms).
    traps = []
    TRAP_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(TRAP_EVENT, 2000)
    
    running = True
    while running:
        clock.tick(FPS)
        
        # Process events (trap timer and window close)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == TRAP_EVENT:
                # Add a new trap at a random location not on snake, fruit, or another trap.
                trap_pos = get_random_trap(snake, fruit, traps)
                traps.append(trap_pos)
        
        head = snake[0]
        # Use A* to compute a path from the snake's head to the fruit.
        path = a_star(head, fruit["pos"], snake)
        if path and len(path) >= 2:
            candidate_move = path[1]
            # Before committing, check connectivity safety.
            if not is_move_safe(candidate_move, snake, fruit):
                candidate_move = get_safe_move(snake, fruit)
            next_move = candidate_move
        else:
            # If no path is found, choose any safe move.
            next_move = get_safe_move(snake, fruit)
        
        # Check for collision with walls.
        if (next_move[0] < 0 or next_move[0] >= GRID_WIDTH or 
            next_move[1] < 0 or next_move[1] >= GRID_HEIGHT):
            print("Collision with wall! Game Over!")
            running = False
            continue

        # Check for trap collision.
        if next_move in traps:
            traps.remove(next_move)  # Remove the trap after collision
            if len(snake) <= PENALTY_SEGMENTS:
                print("Hit a trap and snake is too short! Game Over!")
                running = False
                continue
            else:
                for _ in range(PENALTY_SEGMENTS):
                    snake.pop()  # Remove segments from the tail as a penalty
                print(f"Hit a trap! Snake lost {PENALTY_SEGMENTS} segments.")

        # Check for self-collision.
        if next_move in snake:
            print("Collision with self! Game Over!")
            running = False
            continue
        
        # Update the snake's position: add the new head.
        snake.insert(0, next_move)
        if next_move == fruit["pos"]:
            # The snake eats the fruit: increase score and generate a new fruit.
            score += fruit["value"]
            fruit = get_random_fruit(snake)
        else:
            # Normal movement: remove the tail segment.
            snake.pop()
        
        # Draw the game board.
        screen.fill(BLACK)
        
        # Draw the fruit.
        fruit_rect = pygame.Rect(fruit["pos"][0] * CELL_SIZE, fruit["pos"][1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, fruit["color"], fruit_rect)
        
        # Draw the traps.
        for trap in traps:
            trap_rect = pygame.Rect(trap[0] * CELL_SIZE, trap[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, ORANGE, trap_rect)
        
        # Draw the snake.
        for segment in snake:
            segment_rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GREEN, segment_rect)
        
        # Render and display the score.
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (5, 5))
        
        # Optionally, draw grid lines for clarity:
        # for x in range(0, WINDOW_WIDTH, CELL_SIZE):
        #     pygame.draw.line(screen, WHITE, (x, 0), (x, WINDOW_HEIGHT))
        # for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
        #     pygame.draw.line(screen, WHITE, (0, y), (WINDOW_WIDTH, y))
        
        pygame.display.update()
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
