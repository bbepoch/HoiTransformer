# HOI Transformer
Code for CVPR 2021 accepted paper [End-to-End Human Object Interaction Detection with HOI Transformer](https://arxiv.org/abs/2103.04503).

This method also won 2nd Place Award in HOI Challenge in [Person In Context](http://www.picdataset.com/challenge/leaderboard/pic2021) in CVPR Workshop 2021.

<div align="center">
  <img src="data/architecture.png" width="900px" />
</div>


## Reproduction

We recomend you to setup in the following steps:

1.Clone the repo.
```
git clone https://github.com/bbepoch/HoiTransformer.git
```

2.Download the MS-COCO pretrained [DETR](https://github.com/facebookresearch/detr) model.
```bash
cd data/detr_coco && bash download_model.sh
```

3.Download the annotation files for HICO-DET, V-COCO and HOI-A.
```bash
cd data && bash download_annotations.sh
```

4.Download the image files for [HICO-DET](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk), [V-COCO](https://cocodataset.org/#download) and [HOI-A](https://drive.google.com/drive/folders/15xrIt-biSmE9hEJ2W6lWlUmdDmhatjKt). Instead, we provide a [script](data/download_images.sh) to get all of them. A required directory structure is:

        HoiTransformer/
        ├── data/
        │   ├── detr_coco/
        │   ├── hico/
        │   │   ├── eval/
        │   │   └── images/
        │   │       ├── train2015/
        │   │       └── test2015/
        │   ├── hoia/
        │   │   ├── eval/
        │   │   └── images/
        │   │       ├── trainval/
        │   │       └── test/
        │   └── vcoco/
        │       ├── eval/
        │       └── images/
        │           ├── train2014/
        │           └── val2014/
        ├── datasets/
        ├── models/
        ├── tools/
        ├── util/
        ├── engin.py
        ├── main.py
        └── test.py

5.OPTIONAL SETTINGS. When the above subdirectories in 'data' are all ready, you can train a model on any one of the three benchmarks. But before that, we highly recommend you to move the whole folder 'data' to another place on your computer, e.g. '/home/hoi/data', and only put a soft link named 'data' under 'HoiTransformer'.
```bash
# Optional but recommended to separate data from code.
mv data /home/hoi/
ln -s /home/hoi/data data
```

6.Train a model.
```
# Train on HICO-DET.
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --epochs=150 --lr_drop=110 --dataset_file=hico --batch_size=2 --backbone=resnet50
# Train on HOI-A.
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --epochs=150 --lr_drop=110 --dataset_file=hoia --batch_size=2 --backbone=resnet50
# Train on V-COCO.
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --epochs=150 --lr_drop=110 --dataset_file=vcoco --batch_size=2 --backbone=resnet50
```

7.Test a [model](https://drive.google.com/drive/folders/1RY_4rrUuFzlTfFp5IVTNauB0-Sd0fphW?usp=sharing).
```
python3 test.py --backbone=resnet50 --batch_size=1 --dataset_file=hico --log_dir=./ --model_path=your_model_path

```


## Annotations
We propose a new annotation format 'ODGT' which is much easier to understand, and we have provided annotation files for all the existing benchmarks, i.e. HICO-DET, HOI-A, V-COCO, so just use them. The core structure of ODGT format is:
```json
{
    file_name: XXX.jpg,
    width: image width,
    height: image height,
    gtboxes: [
        {
            box: [x, y, w, h],
            tag: object category name,
        },
        ...
    ],
    hoi: [
        {
            subject_id: human box index in gtboxes,
            object_id: object_box index in gtboxes,
            interaction: hoi category name,
        },
        ...
    ],
}
```


## Citation
```
@inproceedings{zou2021_hoitrans,
  author = {Zou, Cheng and Wang, Bohan and Hu, Yue and Liu, Junqi and Wu, Qian and Zhao, Yu and Li, Boxun and Zhang, Chenguang and Zhang, Chi and Wei, Yichen and Sun, Jian},
  title = {End-to-End Human Object Interaction Detection with HOI Transformer},
  booktitle={CVPR},
  year = {2021},
}
```


## Acknowledgement
We sincerely thank all previous works, especially [DETR](https://github.com/facebookresearch/detr), [PPDM](https://github.com/YueLiao/PPDM), [iCAN](https://github.com/vt-vl-lab/iCAN), for some of the codes are built upon them.

