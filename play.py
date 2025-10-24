import argparse
import time
from snake import SnakeGame
from agent import Agent

def play(speed: int = 20, num_games: int = 1, model_path: str = 'model.pth',
         infinite: bool = False, delay: float = 1.0) -> None:
    agent = Agent()

    # Disable exploration - use only learned policy
    agent.n_games = 1000  # Set high enough to make epsilon=0

    try:
        agent.load_model(model_path)
        print(f'Model loaded from {model_path}')
    except FileNotFoundError:
        print(f'Error: Model file {model_path} not found.')
        print('Please train a model first using train.py')
        return
    except RuntimeError as e:
        print(f'Error loading model: {e}')
        print('The model architecture may have changed. Please retrain using train.py')
        return

    game = SnakeGame(speed=speed, headless=False)
    total_score = 0
    game_num = 0
    best_score = 0

    print(f"{'Playing infinitely' if infinite else f'Playing {num_games} game(s)'}")
    print("Close the window to stop.\n")

    try:
        while True:
            state = game.reset()
            done = False

            while not done:
                action = agent.get_action(state)
                _, done, score = game.play_step(action)
                state = game.get_state()

            game_num += 1
            total_score += score
            best_score = max(best_score, score)

            if infinite:
                print(f'Game {game_num} - Score: {score} - Avg: {total_score / game_num:.2f} - Best: {best_score}')
            else:
                print(f'Game {game_num}/{num_games} - Score: {score}')

            # Stop if we've reached the target number of games (unless infinite)
            if not infinite and game_num >= num_games:
                break

            # Small delay between games
            time.sleep(delay)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    game.cleanup()

    if game_num > 1:
        print(f'\nTotal Games: {game_num}')
        print(f'Average Score: {total_score / game_num:.2f}')
        print(f'Best Score: {best_score}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play Snake with trained AI model')
    parser.add_argument('--speed', type=int, default=20,
                        help='Game speed (default: 20)')
    parser.add_argument('--num-games', type=int, default=1,
                        help='Number of games to play (default: 1)')
    parser.add_argument('--model', type=str, default='model.pth',
                        help='Path to model file (default: model.pth)')
    parser.add_argument('--infinite', action='store_true',
                        help='Play infinitely (ignores --num-games)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay in seconds between games (default: 1.0)')

    args = parser.parse_args()

    play(speed=args.speed, num_games=args.num_games, model_path=args.model,
         infinite=args.infinite, delay=args.delay)
