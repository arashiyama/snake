import pygame
import random
import heapq
import sys

# -------- Configuration --------
CELL_SIZE       = 20    # Pixel size of one grid cell
GRID_WIDTH      = 20    # Number of cells horizontally
GRID_HEIGHT     = 20    # Number of cells vertically
WINDOW_WIDTH    = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT   = GRID_HEIGHT * CELL_SIZE
FPS             = 10    # Game speed (frames per second)

# Colors (R, G, B)
BLACK    = (0, 0, 0)
GREEN    = (0, 255, 0)
RED      = (255, 0, 0)
BLUE     = (0, 0, 255)
YELLOW   = (255, 255, 0)
PURPLE   = (128, 0, 128)  # Traps are purple
WHITE    = (255, 255, 255)

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
    """Return neighboring positions (up, down, left, right) for a given cell."""
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
            # Reconstruct path from start to goal.
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
                
    return None  # No valid path found.

def is_move_safe(new_head, snake, fruit, traps):
    """
    Simulate the snake's move and check whether the new head position will
    eventually allow the snake to reach its tail (i.e. not trap itself).
    Traps are treated as obstacles.
    """
    if new_head in traps:
        return False

    # Simulate new snake state.
    if new_head == fruit["pos"]:
        new_snake = [new_head] + snake[:]  # Snake grows.
    else:
        new_snake = [new_head] + snake[:-1]  # Normal move.
    
    # For connectivity, treat all segments except the tail as obstacles,
    # and also treat traps as obstacles.
    obstacles = set(new_snake[:-1])
    obstacles.update(traps)
    target = new_snake[-1]  # The tail cell we want to reach.

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

def flood_fill_count(start, snake, traps):
    """
    Performs a flood fill from the given start position counting the number
    of reachable cells. Obstacles include the snake's body and the traps.
    """
    obstacles = set(snake)
    obstacles.update(traps)
    visited = set()
    queue = [start]
    count = 0
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        count += 1
        for neighbor in get_neighbors(current):
            if neighbor in visited:
                continue
            x, y = neighbor
            if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
                continue
            if neighbor in obstacles:
                continue
            queue.append(neighbor)
    return count

def get_safe_move(snake, fruit, traps, recent_heads):
    """
    Evaluates the four possible moves from the snake's head.
    For each candidate that is immediately safe (no wall, snake, or trap collision)
    and passes the connectivity test, compute the flood-fill area (free space).
    
    Additionally, if a candidate move lands on one of the snake's recent head positions,
    a penalty is subtracted from its free space value to discourage repeating the same path.
    
    Returns the move that maximizes the adjusted free space available.
    """
    head = snake[0]
    candidate_moves = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    PENALTY_VALUE = 100  # Adjust this value as needed
    
    for dx, dy in directions:
        new_head = (head[0] + dx, head[1] + dy)
        # Check boundaries.
        if new_head[0] < 0 or new_head[0] >= GRID_WIDTH or new_head[1] < 0 or new_head[1] >= GRID_HEIGHT:
            continue
        # Check immediate collision with snake.
        if new_head in snake:
            continue
        # Avoid traps.
        if new_head in traps:
            continue
        # Check connectivity safety.
        if not is_move_safe(new_head, snake, fruit, traps):
            continue
        # Compute available free area from new_head.
        area = flood_fill_count(new_head, snake, traps)
        # Apply a penalty if this move repeats a recent head position.
        if new_head in recent_heads:
            area -= PENALTY_VALUE
        candidate_moves.append((new_head, area))
    
    if candidate_moves:
        # Choose the move that maximizes the (adjusted) flood-fill count.
        candidate_moves.sort(key=lambda x: x[1], reverse=True)
        return candidate_moves[0][0]
    
    # If no candidate passes the connectivity test, return any immediate safe move.
    for dx, dy in directions:
        new_head = (head[0] + dx, head[1] + dy)
        if new_head[0] < 0 or new_head[0] >= GRID_WIDTH or new_head[1] < 0 or new_head[1] >= GRID_HEIGHT:
            continue
        if new_head in snake:
            continue
        return new_head
    return head  # No safe move found; this will result in collision.

# -------- Main Game Loop --------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Self-Playing Snake Game with Purple Traps & Cycle Avoidance")
    clock = pygame.time.Clock()
    
    # Set up a font for displaying the score.
    font = pygame.font.SysFont("arial", 20)
    
    # Initialize the snake in the center.
    snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
    
    # Generate the first fruit.
    fruit = get_random_fruit(snake)
    
    score = 0  # Initialize score.
    
    # Initialize trap storage and timer event.
    # Two traps will be added every 1000 ms.
    traps = []
    TRAP_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(TRAP_EVENT, 1000)
    
    # Track recent head positions to discourage repeated cycles.
    recent_heads = []
    RECENT_LIMIT = 10  # Number of recent head positions to remember.
    
    running = True
    while running:
        clock.tick(FPS)
        
        # Process events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == TRAP_EVENT:
                # Add two traps at random locations (avoiding snake, fruit, or existing traps).
                for _ in range(2):
                    trap_pos = get_random_trap(snake, fruit, traps)
                    traps.append(trap_pos)
        
        head = snake[0]
        # Use A* to compute a path from the snake's head to the fruit.
        path = a_star(head, fruit["pos"], snake)
        if path and len(path) >= 2:
            candidate_move = path[1]
            if not is_move_safe(candidate_move, snake, fruit, traps):
                candidate_move = get_safe_move(snake, fruit, traps, recent_heads)
            next_move = candidate_move
        else:
            next_move = get_safe_move(snake, fruit, traps, recent_heads)
        
        # Check for collision with walls.
        if (next_move[0] < 0 or next_move[0] >= GRID_WIDTH or 
            next_move[1] < 0 or next_move[1] >= GRID_HEIGHT):
            print("Collision with wall! Game Over!")
            running = False
            continue

        # Check for trap collision.
        if next_move in traps:
            traps.remove(next_move)  # Remove the trap that was hit.
            # Calculate penalty: snake loses half of its current length (rounded down).
            penalty = len(snake) // 2
            if len(snake) <= penalty or penalty < 1:
                print("Hit a trap and lost too much of your body! Game Over!")
                running = False
                continue
            else:
                for _ in range(penalty):
                    if len(snake) > 0:
                        snake.pop()
                print(f"Hit a trap! Snake lost {penalty} segments (half its length).")
        
        # Check for self-collision.
        if next_move in snake:
            print("Collision with self! Game Over!")
            running = False
            continue
        
        # Update the snake's position.
        snake.insert(0, next_move)
        if next_move == fruit["pos"]:
            # Snake eats the fruit.
            score += fruit["value"]
            fruit = get_random_fruit(snake)
        else:
            snake.pop()
        
        # Update recent head positions.
        recent_heads.append(snake[0])
        if len(recent_heads) > RECENT_LIMIT:
            recent_heads.pop(0)
        
        # Draw everything.
        screen.fill(BLACK)
        
        # Draw fruit.
        fruit_rect = pygame.Rect(fruit["pos"][0] * CELL_SIZE, fruit["pos"][1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, fruit["color"], fruit_rect)
        
        # Draw traps.
        for trap in traps:
            trap_rect = pygame.Rect(trap[0] * CELL_SIZE, trap[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, PURPLE, trap_rect)
        
        # Draw snake.
        for segment in snake:
            segment_rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GREEN, segment_rect)
        
        # Render score.
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (5, 5))
        
        pygame.display.update()
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
