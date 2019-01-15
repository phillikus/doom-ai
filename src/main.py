from doom.game import play
from doom.runners import run_basic, run_deadly_corridor
from utils.args import parse_arguments
import argparse


def main():
    game_args = parse_arguments()
    play(game_args)
        
if __name__ == '__main__':
    main()
