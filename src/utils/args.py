import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Doom parameters')

    parser.add_argument("--scenario", type=str, default="my_way_home", help="Doom scenario")
    parser.add_argument("--lr", type=float, default=0.0001, help="Loss reduction")
    parser.add_argument("--gamma", type=float, default=0.099, help="Gamma")
    parser.add_argument("--tau", type=float, default=1, help="Tau")
    parser.add_argument("--seed", type=float, default=1, help="Seed")
    parser.add_argument("--num_processes", type=int, default=6, help="Number of processes for parallel algorithms")
    parser.add_argument("--num_steps", type=int, default=20, help="Steps for training")
    parser.add_argument("--max_episode_length", type=int, default=10000, help="Maximum episode length")
    parser.add_argument("--num_actions", type=int, default=3, help="Number of actions")
    parser.add_argument("--model", type=str, default='dqn', help="Model to use for training the AI")
    parser.add_argument("--num_updates", type=int, default=100, help="Number of updates")

    game_args, _ = parser.parse_known_args()
    return game_args
