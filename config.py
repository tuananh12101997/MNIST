import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataroot', type=str, default='./data')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--device', type=str, default='cpu')
    
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--scheduler_milestones', type=list, default=[50, 100])
    parser.add_argument('--scheduler_lambda', type=float, default=0.1)
    
    parser.add_argument('--n_iters', type=int, default=120)
    
    return parser