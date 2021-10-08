# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import os
#robfrom google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog


#Step 1: Register dataset
from detectron2.data.datasets import register_coco_instances
register_coco_instances("aquarium_train", {}, "data/aquarium/train/_annotations.coco.json", "data/aquarium/train")
register_coco_instances("aquarium_val", {}, "data/aquarium/valid/_annotations.coco.json", "data/aquarium/valid")
register_coco_instances("aquarium_test", {}, "data/aquarium/test/_annotations.coco.json", "data/aquarium/test")

MetadataCatalog.get("aquarium_train").thing_classes = ["creatures", "fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]


# simple test to visualize ground truth
if False:
    # visualize training data
    my_dataset_train_metadata = MetadataCatalog.get("aquarium_train")
    dataset_dicts = DatasetCatalog.get("aquarium_train")

    import random
    from detectron2.utils.visualizer import Visualizer

    d = random.sample(dataset_dicts, 1)
        
    img = cv2.imread(d[0]["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d[0])
    #cv2_imshow(vis.get_image()[:, :, ::-1])
    cv2.imwrite(os.path.join("output/" + "temp.png"), vis.get_image()[:, :, ::-1])


#-----------------------------------------------------------------------------
#We are importing our own Trainer Module here to use the COCO validation evaluation during training. Otherwise no validation eval occurs.
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
if False:
    #from .detectron2.tools.train_net import Trainer
    #from detectron2.engine import DefaultTrainer
    # select from modelzoo here: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#coco-object-detection-baselines

    from detectron2.config import get_cfg
    #from detectron2.evaluation.coco_evaluation import COCOEvaluator

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("aquarium_train",)
    cfg.DATASETS.TEST = ("aquarium_val",)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001

    cfg.SOLVER.WARMUP_ITERS = 300
    cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = []#(1000, 1500) ## do not decay learning rate
    #cfg.SOLVER.GAMMA = 0.05

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9 #your number of classes + 1
    cfg.TEST.EVAL_PERIOD = 500

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
if True:
    #test evaluation
    from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset

    #==============================
    #==============================
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("aquarium_train",)
    cfg.DATASETS.TEST = ("aquarium_val",)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001

    cfg.SOLVER.WARMUP_ITERS = 300
    cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = []#(1000, 1500) ## do not decay learning rate
    #cfg.SOLVER.GAMMA = 0.05

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9 #your number of classes + 1
    cfg.TEST.EVAL_PERIOD = 500

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    #==============================
    #==============================

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("aquarium_test", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "aquarium_test")
    inference_on_dataset(trainer.model, val_loader, evaluator)
#-----------------------------------------------------------------------------