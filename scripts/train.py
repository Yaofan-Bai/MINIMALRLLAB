import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.trainer import train
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="从checkpoint继续训练")
    parser.add_argument("--checkpoint", default="ppo_checkpoint.pth", help="checkpoint路径")
    args = parser.parse_args()
    train(args.resume, args.checkpoint)
