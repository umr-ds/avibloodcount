import torch

from adet.config import get_cfg
from detectron2.engine import DefaultPredictor


class BatchPredictor(DefaultPredictor):
    """Run the model on a list of images."""

    def __call__(self, original_images):
        """Run d2 on a list of images.

        Args:
            original_images (list): BGR images cropped with 512*384px
        """
        images = []
        for original_image in original_images:
            if self.input_format == "RGB":
                # the model expects BGR inputs
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to(self.cfg.MODEL.DEVICE)
            inputs = {"image": image, "height": height, "width": width}
            images.append(inputs)
        with torch.no_grad():
            predictions = self.model(images)
        return predictions


class DetModel:

    def __init__(self, cfg, gpu, model, thresh):
        # load config
        self.cfg = get_cfg()
        self.cfg.merge_from_file(cfg)
        self.cfg.MODEL.WEIGHTS = model
        self.cfg.MODEL.DEVICE = "cpu" if gpu is None else "cuda"
        self.cfg.MODEL.BiFPN.NORM = "BN" if gpu is None else "SyncBN"

        # initialize predictor
        self.predictor = BatchPredictor(self.cfg)
        self.thresh = thresh

        self.class_ids = {"1": "Erythrocyte (with nucleus)", "2": "Eosinophil", "3": "Lymphocyte", "4": "Heterophil",
                          "5": "Erythrocyte (cell) with parasite", "6": "Monocyte", "7": "Basophil"}
        self.counts = {v: 0 for v in self.class_ids.values()}
        self.counts.update({"Countable tiles": 0})

    def predict(self, images):
        outputs = self.predictor(images)
        if not outputs:
            return None
        for output in outputs:
            instances = output["instances"]
            for i, box in enumerate(instances.pred_boxes):
                score = instances.scores.to("cpu").numpy()[i]
                pred = instances.pred_classes.to("cpu").numpy()[i]
                if int(pred) + 1 in range(1, 8) and float(score) >= self.thresh:
                    self.counts[self.class_ids[str(int(pred) + 1)]] += 1
            self.counts['Countable tiles'] += 1

    def reset_counter(self):
        self.counts = {v: 0 for v in self.class_ids.values()}
        self.counts.update({"Countable tiles": 0})
