import pygame
import random
import heapq
import sys

# -------- Configuration --------
CELL_SIZE   = 20      # Pixel size of one grid cell
GRID_WIDTH  = 20      # Number of cells horizontally
GRID_HEIGHT = 20      # Number of cells vertically
WINDOW_WIDTH  = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 10              # Game speed (frames per second)

# Colors (R, G, B)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED   = (255, 0, 0)
BLUE  = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

# -------- Fruit Generation --------
def get_random_fruit(snake):
    """
    Returns a dictionary representing a fruit with a random type,
    color, and point value that is not placed on the snake.
    
    Fruit types:
      - Red fruit:   10 points
      - Blue fruit:  20 points
      - Yellow fruit: 5 points
    """
    fruit_types = [
        {"color": RED, "value": 10},
        {"color": BLUE, "value": 20},
        {"color": YELLOW, "value": 5}
    ]
    fruit_type = random.choice(fruit_types)
    
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in snake:
            break

    return {"pos": pos, "color": fruit_type["color"], "value": fruit_type["value"]}

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
    
    The snake's body (except the tail) is considered an obstacle.
    """
    # Treat all snake segments except the tail as obstacles
    obstacles = set(snake[:-1]) if len(snake) > 1 else set()
    
    open_set = []
    # Heap items: (f_score, g_score, position)
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
            # Skip neighbors outside the grid
            if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
                continue
            # Skip obstacles (unless neighbor is the goal)
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

def get_safe_move(snake):
    """
    Fallback strategy: from the snake's head, return the first safe neighboring move
    that does not result in an immediate collision.
    """
    head = snake[0]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for dx, dy in directions:
        new_head = (head[0] + dx, head[1] + dy)
        # Check boundaries
        if new_head[0] < 0 or new_head[0] >= GRID_WIDTH or new_head[1] < 0 or new_head[1] >= GRID_HEIGHT:
            continue
        # Check collision with snake's body
        if new_head in snake:
            continue
        return new_head
    return head  # No safe move found; return head (will cause collision)

# -------- Main Game Loop --------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Self-Playing Snake Game with Varied Fruits")
    clock = pygame.time.Clock()
    
    # Set up a font for displaying the score.
    font = pygame.font.SysFont("arial", 20)
    
    # Initialize the snake at the center of the grid.
    snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
    
    # Generate the first fruit.
    fruit = get_random_fruit(snake)
    
    score = 0  # Initialize score
    
    running = True
    while running:
        clock.tick(FPS)
        
        # Handle window events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        head = snake[0]
        # Use A* to compute a path from the snake's head to the fruit.
        path = a_star(head, fruit["pos"], snake)
        if path and len(path) >= 2:
            # The next move is the second element in the path (first is the current head)
            next_move = path[1]
        else:
            # If no path is found, choose any safe move.
            next_move = get_safe_move(snake)
        
        # Check for collision with walls.
        if (next_move[0] < 0 or next_move[0] >= GRID_WIDTH or 
            next_move[1] < 0 or next_move[1] >= GRID_HEIGHT):
            print("Collision with wall! Game Over!")
            running = False
            continue
        
        # Check for self-collision.
        if next_move in snake:
            print("Collision with self! Game Over!")
            running = False
            continue
        
        # Move the snake by adding the new head.
        snake.insert(0, next_move)
        if next_move == fruit["pos"]:
            # Snake eats the fruit: increase score by the fruit's value and generate a new fruit.
            score += fruit["value"]
            fruit = get_random_fruit(snake)
        else:
            # Remove the tail segment to simulate movement.
            snake.pop()
        
        # Draw the game board.
        screen.fill(BLACK)
        
        # Draw the fruit.
        fruit_rect = pygame.Rect(fruit["pos"][0] * CELL_SIZE, fruit["pos"][1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, fruit["color"], fruit_rect)
        
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
