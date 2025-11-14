import numpy as np
import onnxruntime as ort

import slideio
from torch.utils.data import Dataset


class VsiDataset(Dataset):

    # Basic Instantiation
    def __init__(self, vsi_file_path, model_path, thresh):
        self.vsi_file_path = vsi_file_path
        self.model_path = model_path
        self.thresh = thresh
        self.tile_height = 384
        self.tile_width = 512

        scene = slideio.open_slide(self.vsi_file_path, "VSI").get_scene(0)
        self.img_width = scene.size[0]
        self.img_height = scene.size[1]
            
        self.n_horizontal_tiles = self.img_width // self.tile_width
        self.n_vertical_tiles = self.img_height // self.tile_height

        self.sess = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.scene = 0

    #Reading whole VSI file for inference 
    def _init_slideio(self):
        self.scene = slideio.open_slide(self.vsi_file_path, "VSI").get_scene(0)
        
    # Length of the Dataset
    def __len__(self):
        return self.n_vertical_tiles * self.n_horizontal_tiles

    # Fetch an item from the Dataset
    def __getitem__(self, idx):

        if self.scene==0:
            self._init_slideio()
        try:
            vert_pos, horiz_pos = divmod(idx, self.n_horizontal_tiles)
            top = vert_pos * self.tile_height
            left = horiz_pos * self.tile_width
        
            roi = (left, top, self.tile_width, self.tile_height)
    
            #read roi
            tile = self.scene.read_block(roi)            
            tile = tile[np.newaxis, ...]
                        
            # Run model inference
            results_ort = self.sess.run(["Identity:0"], {"input_1:0": tile.astype(np.float32)})            
            model_decision = False
            
            if results_ort[0][0][0] >= self.thresh:
                model_decision = True  
                
            if model_decision:
                return {
                    "tile": tile,
                    "top": top,
                    "left": left,
                }
            else:
                return None
        except Exception as e:
            print(f"Error at tile index {idx}: {e}")
            return None

