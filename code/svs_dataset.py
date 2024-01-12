import numpy as np
import onnxruntime as ort

from openslide import OpenSlide
from torch.utils.data import Dataset


class SvsDataset(Dataset):

    # Basic Instantiation
    def __init__(self, svs_file_path, model_path, thresh):
        self.svs_file = svs_file_path
        self.thresh = thresh
        self.osr = OpenSlide(svs_file_path)
        self.img_width = self.osr.level_dimensions[0][0]
        self.img_height = self.osr.level_dimensions[0][1]
        self.n_horizontal_tiles = self.img_width // 512
        self.n_vertical_tiles = self.img_height // 384
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Length of the Dataset
    def __len__(self):
        return self.n_vertical_tiles * self.n_horizontal_tiles

    # Fetch an item from the Dataset
    def __getitem__(self, idx):
        vert_pos, horiz_pos = divmod(idx, self.n_horizontal_tiles)
        top, left = vert_pos * 384, horiz_pos * 512
        r = (self.osr.read_region((left, top), 0, (512, 384))).convert('RGB')

        # run model
        tile_arr = np.array(r)
        tile = tile_arr[np.newaxis, ...]

        results_ort = self.sess.run(["Identity:0"], {"input_1:0": tile.astype(np.float32)})
        model_decision = False
        if results_ort[0][0][0] >= self.thresh:
            model_decision = True
        r.close()
        if idx == self.__len__():
            self.osr.close()
        if model_decision:
            return {
                "tile": tile,
                "top": top,
                "left": left,
            }
        else:
            return None
