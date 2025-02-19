import argparse
import json
import os
import time
import torch
import warnings

from detector import DetModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from svs_dataset import SvsDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Counting blood cells in avian whole slide blood smear images.',
    )
    parser.add_argument(
        '--path',
        dest='path',
        help='A directory or a single SVS file.',
        default='/data/input/',
        type=str,
    )
    parser.add_argument(
        '--output-dir', '-o',
        dest='output_dir',
        help='Results in JSON format will be saved in this directory.',
        default='/data/output/',
        type=str,
    )
    parser.add_argument(
        '--config', '-c',
        dest='cfg',
        help='Config file for instance segmentation model (YAML format).',
        default="../configs/MS_R_101_BiFPN_3x_sem.yaml",
        type=str
    )
    parser.add_argument(
        '--cls-model',
        dest='cls_model',
        help='Countability classification model in ONNX format.',
        default="/data/models/efficientNet_B0.onnx",
        type=str
    )
    parser.add_argument(
        '--det-model',
        dest='det_model',
        help='Instance segmentation model.',
        default="/data/models/condInst_R101.pth",
        type=str
    )
    parser.add_argument(
        '--countability-thresh', '-t',
        dest='cls_thresh',
        help='Score threshold for the instance segmentation model.',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--detection-thresh', '-d',
        dest='det_thresh',
        help='Score threshold for the instance segmentation model.',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--gpu',
        dest='gpu',
        help='GPU ID to be used or None for CPU-only inference.',
        type=int
    )
    return parser.parse_args()


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def main_loop(svs_files):
    os.makedirs(args.output_dir, exist_ok=True)

    multi_file = len(svs_files) > 1

    model = DetModel(cfg=args.cfg, gpu=args.gpu, model=args.det_model, thresh=args.det_thresh)
    svs_loop = tqdm(svs_files, disable=not multi_file)
    svs_loop.set_description("Overall")
    for filename in svs_loop:
        # loc_start = time.time()
        dataset = SvsDataset(os.path.join(args.path, filename), args.cls_model, args.cls_thresh)
        loader = DataLoader(dataset,
                            batch_size=8,
                            shuffle=False,
                            num_workers=os.cpu_count() - 1,
                            pin_memory=True,
                            collate_fn=collate_fn)
        tile_loop = tqdm(loader, leave=False)
        tile_loop.set_description("Current file")
        for i, item in enumerate(tile_loop):
            if item:
                images = [img['tile'][..., ::-1].squeeze(axis=0) for img in item]
                model.predict(images)
        counts = model.counts
        model.counts.update({'Overall tiles': len(dataset)})
        del counts[model.class_ids['5']]
        # loc_end = time.time()
        # print(time.strftime("%Hh%Mm%Ss", time.gmtime(loc_end - loc_start)))
        with open(args.output_dir + filename + ".txt", 'w') as f:
            json.dump(counts, f)
        model.reset_counter()


def get_input_files(path):
    if os.path.isfile(path):
        if path.endswith((".svs", ".SVS")):
            svs_files = [path]
        else:
            raise argparse.ArgumentTypeError(f"Expecting input files to be in SVS file format! Found "
                                             f"'{os.path.splitext(path)[-1]}'.")
    elif os.path.isdir(path):
        svs_files = [item for item in os.listdir(path) if item.endswith((".svs", ".SVS"))]
        if not svs_files:
            print(f"Warning: Directory {path} does not contain any SVS file!")
            return
    else:
        raise FileNotFoundError
    return svs_files


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning,
                            message="torch.meshgrid: in an upcoming release, "
                                    "it will be required to pass the indexing argument.")
    args = parse_args()

    # set GPU
    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # avoid problems with shm
    torch.multiprocessing.set_sharing_strategy('file_system')

    # start analysis
    input_files = get_input_files(args.path)
    if input_files:
        start = time.time()
        main_loop(input_files)
        end = time.time()
        print(time.strftime("%Hh%Mm%Ss", time.gmtime(end - start)))
    print("Done.")
