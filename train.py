import argparse
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm
from snake import SnakeGame
from agent import Agent

def plot_scores(scores: List[int], mean_scores: List[float]) -> None:
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score', alpha=0.6)
    plt.plot(mean_scores, label='Average Score')
    plt.ylim(ymin=0)
    plt.legend()
    plt.pause(0.001)

def train(headless: bool = False, speed: int = 40, max_games: int = None,
          target_score: int = None, lr: float = 0.001, gamma: float = 0.9,
          hidden_size: int = 256, memory_size: int = 100_000,
          batch_size: int = 64, train_frequency: int = 4,
          min_samples: int = 100) -> None:
    
    plot_scores_flag = not headless

    if plot_scores_flag:
        plt.ion()

    scores: List[int] = []
    mean_scores: List[float] = []
    total_score = 0
    record = 0

    agent = Agent(input_size=12, hidden_size=hidden_size, output_size=3,
                  lr=lr, gamma=gamma, memory_size=memory_size, batch_size=batch_size)
    game = SnakeGame(speed=speed, headless=headless)

    pbar = tqdm(total=max_games, desc="Training", unit="game",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')

    step_count = 0
    while True:
        state_old = game.get_state()
        action = agent.get_action(state_old)
        reward, done, score = game.play_step(action)
        state_new = game.get_state()

        agent.remember(state_old, action, reward, state_new, done)

        # Train every N steps (only after collecting min_samples)
        step_count += 1
        if len(agent.memory) >= min_samples and step_count % train_frequency == 0:
            agent.train()

        if done:
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score
                agent.save_model()

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)

            epsilon = max(0.0, 1.0 - agent.n_games / agent.epsilon_decay) * 100

            pbar.set_postfix({
                'Score': score,
                'Record': record,
                'Avg': f'{mean_score:.1f}',
                'ε': f'{epsilon:.0f}%'
            })
            pbar.update(1)

            if plot_scores_flag:
                plot_scores(scores, mean_scores)
            if max_games and agent.n_games >= max_games:
                break
            if target_score and score >= target_score:
                break

    pbar.close()
    game.cleanup()

    if plot_scores_flag:
        plt.ioff()
        print("\nTraining complete! Close the plot window to exit.")
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Snake AI using Q-Learning')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI (faster training)')
    parser.add_argument('--speed', type=int, default=40,
                        help='Game speed (default: 40)')
    parser.add_argument('--max-games', type=int, default=None,
                        help='Maximum number of games to train')
    parser.add_argument('--target-score', type=int, default=None,
                        help='Stop training when this score is reached')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='Discount factor (default: 0.9)')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Hidden layer size (default: 256)')
    parser.add_argument('--memory-size', type=int, default=100_000,
                        help='Replay memory size (default: 100,000)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--train-frequency', type=int, default=4,
                        help='Train every N steps (default: 4)')
    parser.add_argument('--min-samples', type=int, default=100,
                        help='Minimum samples before training starts (default: 100)')

    args = parser.parse_args()

    train(headless=args.headless, speed=args.speed, max_games=args.max_games,
          target_score=args.target_score, lr=args.lr, gamma=args.gamma,
          hidden_size=args.hidden_size, memory_size=args.memory_size,
          batch_size=args.batch_size, train_frequency=args.train_frequency,
          min_samples=args.min_samples)
