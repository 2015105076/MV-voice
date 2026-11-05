"""C worker: dispatch one five-fold deep-pipeline run through the locked trainer."""
import argparse

from train import train_run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--views', required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--fold-count', type=int, default=5)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--dropout', type=float, default=.2)
    parser.add_argument('--fusion', default='gated')
    parser.add_argument('--head-lr', type=float, default=.001)
    parser.add_argument('--backbone-multiplier', type=float, default=1.)
    parser.add_argument('--weight-decay', type=float, default=.0001)
    parser.add_argument('--run-id', default=None)
    args = parser.parse_args()
    train_run(args.backbone, args.views, args.fold, args.fold_count, args.seed, args.epochs,
              args.dropout, args.fusion, args.head_lr, args.backbone_multiplier, args.weight_decay, args.run_id)
