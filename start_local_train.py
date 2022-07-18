import ast
import os
import moxing as mox
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_url', type=str, default=None, help='the output path')
parser.add_argument('--data_url',type=str, default='', help='path to datasets only on S3, only need on ModelArts')
parser.add_argument('--config_file', type=str, default='segformer/segformer_mit-b1_512x512_160k_ade20k_segdeformer3.py', help='the path of the config file') #./configs/coco_R_50_FPN_1x_moco.yaml
parser.add_argument('--output', type=str,  default='ckpt_new2.pth', help='destination file name')
parser.add_argument("--data_root", type=str, default='/home/ma-user/work/03data/ADE20k/',
                        help="path to Dataset")
parser.add_argument("--load_from", type=str, default='',
                        help="path to Dataset")
parser.add_argument('--GPU', type=int, default=1)

args, unparsed = parser.parse_known_args()

# ############# preparation stage ####################
args.config_file = os.path.join('configs', args.config_file)
work_dir = '../model_out/'
print(args)
cmd = 'sh tools/dist_train.sh {} {} --work-dir {}'
cmd = cmd.format(args.config_file, args.GPU, work_dir)
print(cmd)
os.system(cmd)
