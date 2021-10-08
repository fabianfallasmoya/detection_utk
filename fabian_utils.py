
if False:
    #from detectron2.data.datasets import register_coco_instances
    #register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
    #register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog, DatasetCatalog

    register_coco_instances("aquarium_train", {}, "data/aquarium/train/_annotations.coco.json", "data/aquarium/train")
    register_coco_instances("aquarium_val", {}, "data/aquarium/valid/_annotations.coco.json", "data/aquarium/valid")
    MetadataCatalog.get("aquarium_train").thing_classes = ["creatures", "fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]

if False:    
    from detectron2.config import get_cfg
    cfg = get_cfg()
    print(cfg.OUTPUT_DIR)


if False:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    import os, json, cv2, random
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    from detectron2.data.datasets import register_coco_instances

    register_coco_instances("aquarium_train", {}, "data/aquarium/train/_annotations.coco.json", "data/aquarium/train")
    register_coco_instances("aquarium_val", {}, "data/aquarium/valid/_annotations.coco.json", "data/aquarium/valid")
    MetadataCatalog.get("aquarium_train").thing_classes = ["creatures", "fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]

    cfg = get_cfg()
    cfg.DATASETS.TRAIN = ("aquarium_train",)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)
    
    from detectron2.utils.visualizer import ColorMode
    from detectron2.data.datasets import load_coco_json
    dataset_dicts = load_coco_json("data/aquarium/valid/_annotations.coco.json", "data/aquarium/valid")

    i = 0
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get('aquarium_train'), scale=0.5, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, str(i) + "_im.png"), out.get_image()[:, :, ::-1])
        i += 1
    
if True:
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    import os, json, cv2, random
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2 import model_zoo

    register_coco_instances("aquarium_train", {}, "data/aquarium/train/_annotations.coco.json", "data/aquarium/train")
    register_coco_instances("aquarium_val", {}, "data/aquarium/valid/_annotations.coco.json", "data/aquarium/valid")
    MetadataCatalog.get("aquarium_train").thing_classes = ["creatures", "fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]

    #cfg = get_cfg()
    #cfg.DATASETS.TRAIN = ("aquarium_train",)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("aquarium_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00125
    cfg.SOLVER.MAX_ITER = 400
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here. (copied from the official detectron2 tutorial)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer = DefaultTrainer(cfg)
    evaluator = COCOEvaluator("aquarium_val", cfg, False, output_dir="./fabian_output/")
    val_loader = build_detection_test_loader(cfg, "aquarium_val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))