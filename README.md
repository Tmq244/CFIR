# MultiturnFashionRetrieval

Codebase for the SIGIR 2021 paper **Conversational Fashion Image Retrieval via Multiturn Natural Language Feedback**.  
Paper: [https://arxiv.org/abs/2106.04128](https://arxiv.org/abs/2106.04128)

This repository is based on the original implementation:
[https://github.com/yfyuan01/MultiturnFashionRetrieval/tree/master](https://github.com/yfyuan01/MultiturnFashionRetrieval/tree/master)

It has been updated for Python 3 compatibility and includes bug fixes to improve reproducibility.

## Repository Structure

```text
CFIR/
├── attr/                  # Fashion attribute annotations
├── Combine/               # Offline scoring export and score fusion scripts
├── data/                  # Train/val conversation JSON files
├── docs/                  # Documentation and guides
├── image_splits/          # Image split metadata
├── images/                # Product image files
├── irbench/               # In-repo IR benchmark module
├── logs/                  # TensorBoard logs
├── Model/                 # Model definitions
├── output_score/          # Offline score outputs
├── preprocess/            # Data loaders, trainer/evaluator, losses
├── repo/                  # Saved checkpoints
├── results/               # Training/evaluation summaries
├── scripts/               # Utility scripts
├── .gitignore             # Git ignore rules
├── image_embedding.pkl    # Image embedding file
├── main.py                # Training entry point
├── README.md              # Project README
└── text_embedding.pkl     # Text embedding file
```

## Environment Setup

### Requirements

- Linux 
- Python 3
- CUDA-capable NVIDIA GPU
- Conda (Anaconda or Miniconda)

### 1) Create environment

```bash
conda create -n cfir python=3.9 -y
conda activate cfir
```

### 2) Install Python dependencies

Install Python dependencies (including PyTorch):

```bash
pip install torch torchvision torchaudio tensorboardX tqdm numpy Pillow easydict hyperopt
```

## Data Preparation

This project uses a multiturn conversational fashion retrieval dataset with three categories:
`dress`, `shirt`, and `toptee`.

- Conversation files are in `data/`
- Attribute files are in `attr/`

Download required assets:

- Images: [Google Drive](https://drive.google.com/file/d/1pivWpO3_vpMLhySmQc9w53i9Tp0ib1lg/view?usp=sharing)
- Image embedding: [image_embedding](https://drive.google.com/file/d/1iHc-XUFTGcgt3udw8iEM0CialEtj2pt8/view?usp=sharing)
- Text embedding: [text_embedding](https://drive.google.com/file/d/19mhuZMQgVprLkEuTu5cBN5meukuGFnjb/view?usp=sharing)

Place files as follows:

- Image files under `images/*.jpg`
- `image_embedding.pkl` in repository root
- `text_embedding.pkl` in repository root

Important path arguments:

- `--data_root` points to your dataset JSON directory (default: `data/`)
- `--image_root` is joined as `os.path.join(image_root, "images/<asin>.jpg")`
  - If your images are at `<repo>/images/*.jpg`, use `--image_root .`

## Training

Training entry point: `main.py`

Recommended baseline command (`combine`):

```bash
python main.py \
  --method combine \
  --target dress \
  --backbone resnet152 \
  --text_method encode \
  --stack_num 3 \
  --fdims 2048 \
  --batch_size 16 \
  --epochs 10 \
  --lr 0.0001 \
  --gpu_id 0 \
  --expr_name combine_dress_r152_sa3_e10 \
  --data_root data/ \
  --image_root .
```

Common model choices:

- `--method combine` (cross-attention model, recommended main setting)
- `--method tirg`
- `--method text-only`
- `--method image-only`

Training outputs:

- `logs/<expr_name>/` TensorBoard logs
- `repo/<expr_name>/best_model.pth` best checkpoint
- `results/<timestamp>_<expr_name>.json` per-epoch summary and best metrics

## Testing / Evaluation

Evaluation is integrated into training and runs automatically at the end of each epoch.

Main retrieval metrics:

- `R@1`, `R@5`, `R@8`, `R@10`, `R@20`, `R@50`
- `R10R50` (average of `R@10` and `R@50`)
- `MRR`

Monitor training/evaluation curves with TensorBoard:

```bash
tensorboard --logdir logs/ --port 6006
```

## Optional: Offline Scoring and Fusion

After training, you can export offline scores for fusion:

```bash
python Combine/get_score.py
```

Then optimize weights across multiple model outputs:

```bash
python Combine/optimize_score.py
```

## Citation

If you find this useful, please cite:

```bibtex
@inproceedings{10.1145/3404835.3462881,
author = {Yuan, Yifei and Lam, Wai},
title = {Conversational Fashion Image Retrieval via Multiturn Natural Language Feedback},
year = {2021},
booktitle = {Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
}
```

