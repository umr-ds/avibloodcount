import numpy as np
import onnxruntime as ort

from pylibCZIrw import czi as pyczi
from torch.utils.data import Dataset

class CziDataset(Dataset):

    # Basic Instantiation
    def __init__(self, czi_file_path, model_path, thresh):
        self.czi_file_path = czi_file_path
        self.thresh = thresh
        self.tile_height = 384
        self.tile_width = 512
        with pyczi.open_czi(self.czi_file_path) as czi:
            bbox = czi.total_bounding_rectangle
            self.origin_x = bbox.x
            self.origin_y = bbox.y
            self.img_width = bbox.w
            self.img_height = bbox.h    
            
        self.n_horizontal_tiles = self.img_width // self.tile_width
        self.n_vertical_tiles = self.img_height // self.tile_height
        
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Length of the Dataset
    def __len__(self):
        return self.n_vertical_tiles * self.n_horizontal_tiles

    # Fetch an item from the Dataset
    def __getitem__(self, idx):

        try:
            vert_pos, horiz_pos = divmod(idx, self.n_horizontal_tiles)
            top = self.origin_y + vert_pos * self.tile_height
            left = self.origin_x + horiz_pos * self.tile_width
        
            roi = (left, top, self.tile_width, self.tile_height)
            with pyczi.open_czi(self.czi_file_path) as czidoc:
                tile = czidoc.read(roi=roi, plane={"C": 0, "Z": 3})
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

