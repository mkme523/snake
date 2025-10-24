# Snake AI with Q-Learning

A Python/PyTorch implementation of the classic Snake game with Q-Learning reinforcement learning agent.

## Setup

### Prerequisites
- Python 3.13+

### Installing Python 3.13

#### macOS
```bash
brew install python@3.13
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13 python3.13-pip python3.13-venv
```

#### Windows
```bash
winget install Python.Python.3.13
```

#### Verify Installation
```bash
python3.13 --version
```

### Installing Dependencies
```bash
python3.13 -m pip install -r requirements.txt
```

## Usage

### Training the Model

#### Basic Training
```bash
python3.13 train.py
```

#### Headless Training
```bash
python3.13 train.py --headless
```

#### Training with Custom Parameters
```bash
python3.13 train.py --headless --speed 1000 --max-games 500
```

#### Available Training Options
- `--headless`: Run without GUI for faster training
- `--speed`: Game speed in FPS (default: 40)
- `--max-games`: Maximum number of games to train
- `--target-score`: Stop training when this score is reached
- `--lr`: Learning rate (default: 0.001)
- `--gamma`: Discount factor for Q-learning (default: 0.9)
- `--hidden-size`: Neural network hidden layer size (default: 256)
- `--memory-size`: Replay memory size (default: 100,000)
- `--batch-size`: Training batch size (default: 64)
- `--train-frequency`: Train every N steps (default: 4)
- `--min-samples`: Minimum samples before training starts (default: 100)

#### Example Training Commands
```bash
python3.13 train.py --headless --speed 1000 --max-games 1000

python3.13 train.py --headless --target-score 50

python3.13 train.py --lr 0.0005 --gamma 0.95 --hidden-size 512
```

### Playing with Trained Model

#### Basic Play
```bash
python3.13 play.py
```

#### Play Multiple Games
```bash
python3.13 play.py --num-games 10
```

#### Play with Custom Speed
```bash
python3.13 play.py --speed 10
```

#### Available Play Options
- `--speed`: Game speed in FPS (default: 20)
- `--num-games`: Number of games to play (default: 1)
- `--model`: Path to model file (default: model.pth)
- `--infinite`: Play infinitely (ignores --num-games)
- `--delay`: Delay in seconds between games (default: 1.0)

#### Example Play Commands
```bash
# Play 5 games
python3.13 play.py --num-games 5

# Play infinitely (Ctrl+C to stop)
python3.13 play.py --infinite

# Play fast with no delay
python3.13 play.py --infinite --delay 0 --speed 100

# Use a different model
python3.13 play.py --model my_model.pth
```

## How It Works

### Q-Learning Agent
The agent uses Deep Q-Learning with experience replay to learn optimal snake behavior. The neural network takes a 12-dimensional state vector as input:
- 3 danger indicators (straight, right, left)
- 4 direction indicators (left, right, up, down)
- 4 food location indicators (left, right, up, down relative to head)
- 1 normalized taxicab distance to food

The agent outputs Q-values for 3 possible actions:
- Continue straight
- Turn right
- Turn left

### Training Process
1. The agent explores using epsilon-greedy strategy
2. Experiences are stored in replay memory
3. The network is trained on batches of past experiences
4. The best model (highest score) is automatically saved

### Reward Structure
- +10 for eating food
- -10 for collision (wall or self)
- +0.1 for moving closer to food (encourages goal-directed behavior)
- -0.1 for moving away from food (discourages loops and wandering)

## Files

- `snake.py`: Snake game environment implementation
- `agent.py`: Q-Learning neural network and agent
- `train.py`: Training script with customizable parameters
- `play.py`: Script to watch trained agent play
- `model.pth`: Saved model weights (created after training)

