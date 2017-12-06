import argparse

from model import *
from utils import *
from PGGAN import *

parse = argparse.ArgumentParser()
parse.add_argument('--is_training',default=False, type=bool,help='')
parse.add_argument('--gpu_option', default=True, type=bool, help='')
parse.add_argument('--target_r', default=128, type=int, help='')
parse.add_argument('--source_r', default=32, type=int, help='')
parse.add_argument('--stablize_kimgs', default=600, type=int, help='')
parse.add_argument('--fade_in_kimgs', default=600, type=int, help='')
parse.add_argument('--latent_size', default=512, type=int, help='')
parse.add_argument('--sampling_ite', default=500, type=int, help='')
parse.add_argument('--sample_dir',default='./sample',type=str, help='')
parse.add_argument('--saving_ite', default=1000, type=int, help='')
parse.add_argument('--checkpoint_dir', default='checkpoint', type=str, help='')
parse.add_argument('--gan', default='lsgan',type=str, help='')
parse.add_argument('--learning_rate', default=0.001, type=float, help='')
parse.add_argument('--beta1', default=0, type=float, help='')
parse.add_argument('--beta2', default=0.99, type=float, help='')
parse.add_argument('--tanh_bool', default=False, type=bool, help='')
def check_dir():
    if not os.path.exists('./sample'):
        os.mkdir('./sample')
    if not os.path.exists('./checkpoint'):
        os.mkdir('./checkpoint')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

if __name__=='__main__':
    check_dir()
    args = parse.parse_args()
    config = {k:v for k,v in args._get_kwargs()}
    # data = 
    sigmoid_at_end = True if config['gan'] == 'lsgan' else False
    #tanh_at_end = True if config['tanh_bool'] else False
    tanh_at_end = False
    print(sigmoid_at_end)
    print(tanh_at_end)
    D = discriminator(num_channels=3, resolution=args.target_r, feature_map_max=512, feature_map_base=8192, sigmoid_at_end=sigmoid_at_end)
    G = generator(num_channels=3, latent_size=512, resolution=args.target_r, feature_map_max=512, feature_map_base=8192, tanh_at_end=tanh_at_end)
    print(G)
    print(D)
    for item in config.keys():
        print(item,':', config[item])
    data = CelebA()
    pggan = PGGAN(D, G, data, config)
    if args.is_training:
        pggan.train()
