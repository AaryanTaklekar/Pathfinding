import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from heapq import heappush, heappop
import os
import platform

# === Parameters ===
GRID_SIZE = 20
CELL_SIZE = 30
WINDOW_SIZE = (GRID_SIZE * CELL_SIZE) 
EPOCHS = 100  
BATCH_SIZE = 32  # Reduced batch size
LEARNING_RATE = 0.0005
HIDDEN_SIZE = 128

# === Colors ===
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)
BLUE = (0, 191, 255)

# === Device Selection Logic ===
def get_device():
    """
    Enhanced device selection logic that automatically chooses the best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA device: {gpu_name}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Check for Apple Silicon (MPS)
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) acceleration")
    else:
        device = torch.device("cpu")
        cpu_name = platform.processor()
        print(f"Using CPU: {cpu_name}")
    
    # Print additional device information
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    
    return device

# Initialize device
device = get_device()

# === Pygame Setup ===
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 100))
pygame.display.set_caption("A* vs RNN Pathfinding (Enhanced)")
font = pygame.font.Font(None, 18)

# === A* Pathfinding ===
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, end):
    open_set = []
    heappush(open_set, (0 + heuristic(start, end), 0, start, [start]))
    visited = set()

    while open_set:
        _, cost, current, path = heappop(open_set)
        if current == end:
            return path
        if current in visited:
            continue
        visited.add(current)

        x, y = current
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx][ny] == 1 and (nx, ny) not in visited:
                new_cost = cost + 1
                heappush(open_set, (new_cost + heuristic((nx, ny), end), new_cost, (nx, ny), path + [(nx, ny)]))

    return None

# === Enhanced RNN Model ===
class PathfindingNet(nn.Module):
    def __init__(self):
        super(PathfindingNet, self).__init__()
        
        # Convolution layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.output_conv = nn.Conv2d(32, 1, kernel_size=1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.sigmoid(self.output_conv(x))
        return x.squeeze(1)

# === Utils ===
def generate_complex_solvable_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    visited = set()
    path = []

    def dfs(x, y):
        if (x, y) in visited or not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return False
        visited.add((x, y))
        path.append((x, y))
        if (x, y) == (GRID_SIZE - 1, GRID_SIZE - 1):
            return True

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(directions)

        for dx, dy in directions:
            if dfs(x + dx, y + dy):
                return True

        path.pop()
        return False

    dfs(0, 0)

    for x, y in path:
        grid[x, y] = 1

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i, j] == 0 and random.random() > 0.7:
                grid[i, j] = 1

    return grid.tolist()

def encode_grid(grid, start, end):
    encoded = np.zeros((3, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    
    # Channel 0: Grid structure
    encoded[0] = np.array(grid, dtype=np.float32)
    
    # Channel 1: Distance from start
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i][j] == 1:
                encoded[1, i, j] = 1 - (heuristic((i, j), start) / (GRID_SIZE * 2))
    
    # Channel 2: Distance to end
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i][j] == 1:
                encoded[2, i, j] = 1 - (heuristic((i, j), end) / (GRID_SIZE * 2))
    
    return encoded

def generate_training_data(num_samples=1000):
    grids = []
    features = []
    targets = []
    
    for _ in range(num_samples):
        grid = generate_complex_solvable_grid()
        start = (0, 0)
        end = (GRID_SIZE - 1, GRID_SIZE - 1)
        grid[start[0]][start[1]] = 1
        grid[end[0]][end[1]] = 1
        
        path = a_star(grid, start, end)
        if path:
            encoded = encode_grid(grid, start, end)
            target = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            for x, y in path:
                target[x, y] = 1.0
            
            grids.append(grid)
            features.append(encoded)
            targets.append(target)
    
    return grids, features, targets

def train_model():
    print("Starting training process...")
    print(f"Training on device: {device}")
    model = PathfindingNet().to(device)
    
    # Check if model has parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    grids, features, targets = generate_training_data(num_samples=10000)
    print(f"Generated {len(grids)} training samples")
    
    # Split into train/val
    split_idx = int(len(grids) * 0.9)
    train_features = features[:split_idx]
    train_targets = targets[:split_idx]
    val_features = features[split_idx:]
    val_targets = targets[split_idx:]
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        # Training
        for i in range(0, len(train_features), BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, len(train_features))
            batch_features = torch.tensor(train_features[i:end_idx], dtype=torch.float32).to(device)
            batch_targets = torch.tensor(train_targets[i:end_idx], dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_features), BATCH_SIZE):
                end_idx = min(i + BATCH_SIZE, len(val_features))
                batch_features = torch.tensor(val_features[i:end_idx], dtype=torch.float32).to(device)
                batch_targets = torch.tensor(val_targets[i:end_idx], dtype=torch.float32).to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        avg_train_loss = epoch_loss / max(1, len(train_features) // BATCH_SIZE)
        avg_val_loss = val_loss / max(1, len(val_features) // BATCH_SIZE)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try:
                # Save model to CPU to ensure compatibility
                model_cpu = PathfindingNet()
                model_cpu.load_state_dict(model.state_dict())
                torch.save(model_cpu.state_dict(), 'best_model.pth')
            except Exception as e:
                print(f"Could not save model: {e}")
    
    return model

# === Inference ===
def predict_rnn(model, grid, end, start=(0, 0)):
    model.eval()
    with torch.no_grad():
        encoded = encode_grid(grid, start, end)
        input_tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_tensor).squeeze().cpu().numpy()
        
        # Mask invalid positions
        output[np.array(grid) == 0] = 0
        
        # Extract path
        path = []
        current = start
        visited = set()
        
        while current != end and len(path) < GRID_SIZE * 3:
            path.append(current)
            visited.add(current)
            
            x, y = current
            candidates = []
            
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 
                    grid[nx][ny] == 1 and (nx, ny) not in visited):
                    score = output[nx, ny]
                    candidates.append(((nx, ny), score))
            
            if not candidates:
                # If stuck, fall back to A*
                a_path = a_star(grid, current, end)
                if a_path and len(a_path) > 1:
                    path.extend(a_path[1:])
                break
            
            # Choose best candidate
            current = max(candidates, key=lambda x: x[1])[0]
        
        if current == end and current not in path:
            path.append(end)
        
        return path, output

# === Pygame Visualization ===
def draw_grid(grid, path=[], start=(0, 0), end=(GRID_SIZE-1, GRID_SIZE-1), buttons=[], 
              nn_scores=None, training_info=None):
    screen.fill(WHITE)
    
    # Draw grid
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color = WHITE if grid[i][j] == 1 else BLACK
            
            # Show neural network confidence
            if nn_scores is not None and grid[i][j] == 1:
                intensity = int(255 * nn_scores[i, j])
                color = (intensity, intensity, 255)
            
            pygame.draw.rect(screen, color, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, GRAY, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
            
            # Display confidence values
            if nn_scores is not None and grid[i][j] == 1:
                text = font.render(f"{nn_scores[i, j]:.2f}", True, BLACK if nn_scores[i, j] > 0.5 else WHITE)
                screen.blit(text, (j * CELL_SIZE + 2, i * CELL_SIZE + 2))
    
    # Draw path
    for idx, (x, y) in enumerate(path):
        pygame.draw.rect(screen, YELLOW, (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Draw start and end
    pygame.draw.rect(screen, GREEN, (start[1] * CELL_SIZE, start[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, RED, (end[1] * CELL_SIZE, end[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Draw buttons
    for btn in buttons:
        pygame.draw.rect(screen, GRAY, btn["rect"])
        label = font.render(btn["text"], True, WHITE)
        screen.blit(label, (btn["rect"].x + 10, btn["rect"].y + 10))
    
    # Display training information
    if training_info:
        info_text = font.render(training_info, True, BLACK)
        screen.blit(info_text, (10, WINDOW_SIZE + 55))
    
    # Display device information
    device_text = font.render(f"Running on: {device}", True, BLACK)
    screen.blit(device_text, (WINDOW_SIZE - 150, WINDOW_SIZE + 55))
    
    pygame.display.update()

def get_clicked_button(pos, buttons):
    for btn in buttons:
        if btn["rect"].collidepoint(pos):
            return btn["action"]
    return None

# === Main Loop ===
def main():
    model = None
    training_info = "Model not trained yet"
    
    grid = generate_complex_solvable_grid()
    start = (0, 0)
    end = (GRID_SIZE - 1, GRID_SIZE - 1)
    grid[start[0]][start[1]] = 1
    grid[end[0]][end[1]] = 1
    
    buttons = [
        {"text": "Run A*", "rect": pygame.Rect(10, WINDOW_SIZE + 10, 80, 40), "action": "a_star"},
        {"text": "Run RNN", "rect": pygame.Rect(100, WINDOW_SIZE + 10, 80, 40), "action": "rnn"},
        {"text": "New Grid", "rect": pygame.Rect(190, WINDOW_SIZE + 10, 100, 40), "action": "regenerate"},
        {"text": "Train RNN", "rect": pygame.Rect(300, WINDOW_SIZE + 10, 100, 40), "action": "train"},
    ]
    
    path = []
    nn_scores = None
    ground_truth = a_star(grid, start, end)
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        clock.tick(30)
        
        draw_grid(grid, path, start, end, buttons, nn_scores, training_info)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                action = get_clicked_button(event.pos, buttons)
                
                if action == "a_star":
                    result = a_star(grid, start, end)
                    if result:
                        path = result
                        nn_scores = None
                        training_info = f"A* path length: {len(path)}"
                
                elif action == "rnn":
                    if model is None:
                        training_info = "Please train the model first!"
                    else:
                        result, scores = predict_rnn(model, grid, end)
                        if result:
                            path = result
                            nn_scores = scores
                            
                            # Calculate accuracy
                            rnn_set = set(path)
                            gt_set = set(ground_truth)
                            intersection = rnn_set & gt_set
                            accuracy = len(intersection) / len(gt_set) if gt_set else 0
                            training_info = f"RNN Accuracy: {accuracy*100:.1f}% Path length: {len(path)}"
                
                elif action == "regenerate":
                    grid = generate_complex_solvable_grid()
                    grid[start[0]][start[1]] = 1
                    grid[end[0]][end[1]] = 1
                    path = []
                    nn_scores = None
                    ground_truth = a_star(grid, start, end)
                    training_info = "New grid generated"
                
                elif action == "train":
                    training_info = "Training in progress..."
                    draw_grid(grid, path, start, end, buttons, nn_scores, training_info)
                    pygame.display.update()
                    
                    model = train_model()
                    training_info = "Training completed!"
    
    pygame.quit()

if __name__ == "__main__":
    main()