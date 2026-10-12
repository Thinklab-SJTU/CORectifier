# ✨ CORectifier

This is the official implementation of our NeurIPS 2026 paper "[**Hierarchically Rectifying Policy Rollouts for Reinforcement Learning in Combinatorial Solvers**](https://openreview.net/forum?id=bpt10w03zk)".

**Authors**: [**Wenzheng Pan**](https://wzever.github.io/) · [**Nuoyan Chen**](https://cny123222.github.io/about/) · [**Jiaxi Liu**](https://ecthelionliu.github.io) · [**Jiale Ma**](https://heatingma.github.io/) · [**Junchi Yan**](https://thinklab.sjtu.edu.cn) | [![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-b31b1b.svg)](paper_preview.pdf) [[paper preview*](paper_preview.pdf)]

*The current PDF is provided for early readers only; please refer to the officially published camera-ready version for the definitive paper.

---
CORectifier gives policy-gradient solvers useful action-level supervision without breaking feasibility. During training, a hierarchical **batch → trajectory → segment** gate selectively replaces a policy action with the feasible successor from an expert tour, then continues the rollout from the executed state. The policy is optimized with the resulting rollout reward and likelihoods; at inference time, the model is used normally with no expert data or rectifier.

![CORectifier overview](assets/fig1.png)

## 🌟 What the paper adds

- **State-consistent guidance:** every replacement is checked against the current feasible state, so sequential constraints remain intact.
- **Three levels of control:** batch, trajectory and contiguous segment gates regulate when and how much expert information enters training.
- **One training recipe across CO:** the release covers Traveling Salesman Problem (TSP), Asymmetric TSP (ATSP), Prize-Collecting TSP (PCTSP), Capacitated Vehicle Routing Problem (CVRP), Knapsack Problem (KP), with TSPLIB and CVRPLIB evaluation entries. More applicable tasks: Job-Shop Scheduling Problem (JSSP), Single Machine Total Weighted Tardiness Problem (SMTWTP), etc. 
- **Better scaling for RL:** the paper reports strong gains through 500-node routing instances while keeping greedy inference lightweight.

## 🧰 Install

Linux, Python 3.8, PyTorch with CUDA and a C/C++ compiler are the tested setup.

```bash
conda create -n corectifier python=3.8
conda activate corectifier
pip install -r requirements.txt
make -C ATSP/utils
make -C CVRP/utils/c_local_search
```

Entries default to `cuda:0`. Set `USE_CUDA=False` in a POMO/MatNet script or `opts.no_cuda=True` for PCTSP to run on CPU.

## 🗂️ Repository map

```text
TSP/ ATSP/ CVRP/ KP/       POMO/MatNet tasks and train/test entry points
PCTSP/                       Attention Model task
weights/                     19 released checkpoints (main rows + CVRPLIB-100)
data/{train,test}/          ignored datasets; keep the .gitkeep files
assets/fig1.png              paper overview figure
```

The task layout follows the original research code. Every resource path in an entry is relative to the repository and is declared in explicit dictionaries such as `TEST_PATH_DICT`, `TRAIN_FOLDER_DICT`, `VAL_PATH_DICT` and `WEIGHT_DICT`.

## 📦 Data

Datasets are intentionally excluded from Git. Keep `data/train/.gitkeep` and `data/test/.gitkeep`, then place files using the paths declared by the selected task script. The public ML4CO sources are [test archive](https://drive.google.com/file/d/1tLQBUBAMJLIcW3KtS5gSXzuuKmdG9VOD/view?usp=share_link) and [training dataset](https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL/tree/main/train_dataset). The repository includes one tiny labeled training example per supported scale for a smoke test; replace it with full shards for training. PCTSP also needs its matching `*_val.txt` and `*_cost.txt` files.

Synthetic main-task scales match the released checkpoints: TSP **50/100/500**, ATSP **50/100/200/500**, CVRP **50/100/200/500**, KP **50/100/200/500**, and PCTSP **20/50/100**. Larger instances are available only through the TSPLIB/CVRPLIB extension files.

## 🚀 Evaluate

```bash
python TSP/test_tsp.py
python ATSP/test_atsp.py
python CVRP/test_cvrp.py
python KP/test_kp.py
python PCTSP/test/test_pctsp.py
```

Defaults use the smallest supported checkpoint and its **full test dataset**, augmentation **16**, CUDA device **0**, TSP/ATSP **2-opt**, and CVRP **classic local search**. `NUM_INSTANCES_DICT` selects `num_instances` (PCTSP `opts.val_size`) for the chosen scale: TSP-50/100 **1,280**, TSP-500 **128**; ATSP-50/100 **2,500**, ATSP-200/500 **100**; CVRP-50/100 **10,000**, CVRP-200/500 **100**; KP **1,280** per scale; PCTSP **10,000** per scale. Change `TEST_DATA` (PCTSP `NODES_NUM`) to select a scale and set `NUM_AUG`/`aug_factor` to the paper row (e.g., `1`, `16` or `128`) as needed.

For real-world generalization:

```bash
python TSP/test_tsplib.py
python CVRP/test_cvrplib.py
```

The TSPLIB/CVRPLIB scripts enumerate the copied library files and select a compatible released model through their explicit `WEIGHT_DICT`.

The CVRPLIB entry maps the 100-customer model to `cvrp100_cvrplib.pt` (epoch 7192), matching the original evaluation script. It comes from the same synthetic CVRP100 training run as `cvrp100.pt` (epoch 10619), with Rectifier enabled and imitation loss disabled. The separate file preserves the original checkpoint choice for CVRPLIB evaluation; it is not a model trained on CVRPLIB.

## 🏋️ Train

```bash
python TSP/train_tsp.py
python ATSP/train_atsp.py
python CVRP/train_cvrp.py
python KP/train_kp.py
python PCTSP/train_pctsp.py
```

**For best training results, train progressively from smaller to larger instances. Each larger scale should continue from the final checkpoint of the previous scale.** Use `50 → 100 → 500` for TSP, `20 → 50 → 100` for PCTSP, and `50 → 100 → 200 → 500` for ATSP/CVRP/KP. The smallest scale starts without a predecessor checkpoint. At each scale, we recommend a short labeled IL warm-up before continuing with Rectified RL (RRL).

Set `NODES_NUM` in the entry script (`ITEMS_NUM` for KP) to select a scale. RRL epochs, batch size, samples per epoch and initial learning rate are selected automatically from **Appendix E, Tables 14–15**. The separate pre-training column gives the warm-up budget to set manually when running that stage:

| Task | Size | Pre-training epochs | RRL epochs | RRL batch size | Samples / epoch | Initial LR |
| --- | --- | --- | --- | --- | --- | --- |
| TSP | 50 | 200 | 2,000 | 200 | 10,000 | 2e-4 |
| TSP | 100 | 100 | 10,000 | 64 | 10,000 | 1e-4 |
| TSP | 500 | 100 | 1,000 | 2 | 10,000 | 1e-4 |
| ATSP | 50 | 200 | 8,000 | 200 | 10,000 | 4e-4 |
| ATSP | 100 | 100 | 3,000 | 64 | 10,000 | 2e-4 |
| ATSP | 200 | 100 | 3,000 | 8 | 10,000 | 2e-4 |
| ATSP | 500 | 100 | 2,000 | 2 | 10,000 | 5e-5 |
| CVRP | 50 | 1,000 | 12,000 | 200 | 10,000 | 2e-4 |
| CVRP | 100 | 1,000 | 12,000 | 128 | 10,000 | 2e-4 |
| CVRP | 200 | 500 | 8,000 | 16 | 10,000 | 1e-4 |
| CVRP | 500 | 100 | 2,000 | 8 | 10,000 | 1e-4 |
| KP | 50 / 100 | 1,000 | 5,000 | 200 | 10,000 | 1e-4 |
| KP | 200 | 500 | 1,000 | 64 | 10,000 | 5e-5 |
| KP | 500 | 100 | 1,000 | 16 | 10,000 | 5e-5 |
| PCTSP | 20 / 50 / 100 | 200 | 10,000 | 64 | 6,400 | 1e-4 |

Table 14 calls the TSP/ATSP/PCTSP warm-up stage **IL**; Table 15 lists CVRP/KP budgets as **pretrain**. To follow the recommended recipe:

1. **Load the previous scale's final checkpoint.** For TSP/ATSP/CVRP/KP, set `trainer_params['model_load']['enable']=True` and point `path` to the preceding run's final `checkpoint-<epoch>.pt`; set `epoch=0` to run the new scale's full epoch budget. For PCTSP, set `opts.load_path` to the preceding run's final `epoch-<epoch>.pt`, with `opts.resume=None` and `opts.epoch_start=0`.
2. **Warm up with labeled IL.** Keep `USE_LABEL=True` / `opts.use_label=True`. Enable `trainer_params['IL_loss']=True` for TSP/ATSP, `USE_IL=True` for CVRP/KP, or `opts.il_loss=True` for PCTSP. Set `MAX_EPOCHS` / `opts.n_epochs` to the pre-training budget above; adjust the IL batch size to available memory.
3. **Continue with RRL.** Load the warm-up's final checkpoint, disable the IL flag, and restore the selected scale's RRL budget and Rectifier settings. After RRL finishes, use its final checkpoint to initialize the next scale.

ATSP's `only_weight=True` and PCTSP's `load_path` initialize model weights for a new stage. TSP/CVRP/KP also restore optimizer state, so check the loaded learning rate and reset it to the selected scale's initial LR when starting a new stage.

All tasks use Adam with `weight_decay=1e-6`. Defaults use CUDA device 0 and validation augmentation 16. **Rectifier is enabled** (`USE_LABEL=True` or `opts.use_label=True`); IL and checkpoint loading are opt-in through the settings above. TSP/ATSP use 2-opt during validation; CVRP uses local search. Training POMO size is `min(size, 200)`.

Download the full labeled training shards before running these budgets. For a toy smoke test, manually set `MAX_EPOCHS`, `BATCH_SIZE` and `SAMPLES_PER_EPOCH` to 1 (PCTSP: `opts.n_epochs`, `opts.batch_size` and `opts.epoch_size`). PCTSP's `options.py` also selects the same settings and matching data paths from `--graph_size`, while preserving explicit CLI overrides.

## 🎛️ Rectifier knobs

| Setting | Meaning |
| --- | --- |
| `use_label` / `USE_LABEL` | Enable expert-labeled batches during training. |
| `label_prob` | Batch-level probability of choosing labeled data. |
| `instance_prob` | Fraction gate for trajectories/instances selected for rectification. |
| `rect_range=[α, β]` | TSP/ATSP/PCTSP segment fraction sampled for replacement. |
| `act_prob` | CVRP route gate; TSP/ATSP/PCTSP keep it at `1.0` and use their segment masks. |
| `IL_loss` / `opts.il_loss` | Optional imitation objective; disabled by default. |

Appendix defaults are `label_prob=instance_prob=0.1` for TSP/PCTSP/CVRP/KP and `0.5` for ATSP; `rect_range=[0.1, 0.2]` for TSP/PCTSP and `[0.8, 1.0]` for ATSP; CVRP uses `act_prob=0.1`. Independent cosine schedules use `T_max=10/20/40/40/40` for batch probability, instance probability, segment lower/upper bounds and CVRP route probability, where applicable.

The reference tour is used only while training. The executed action is rescored by the policy, preserving the original RL objective and feasibility checks.

## 📚 Acknowledgements

The implementations build on [POMO](https://github.com/yd-kwon/POMO), [MatNet](https://github.com/yd-kwon/MatNet), [AM](https://github.com/wouterkool/attention-learn-to-route), [UniCO](https://github.com/Thinklab-SJTU/UniCO), and [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit).

## ✍🏻 Citation
```
@inproceedings{
CORectifier,
title={CORectifier: Hierarchically Rectifying Policy Rollouts for Reinforcement Learning in Combinatorial Solvers},
author={Wenzheng Pan and Nuoyan Chen and Jiaxi Liu and Jiale Ma and Junchi Yan},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026}
}
```
Please use the BibTeX entry exported from OpenReview after the camera-ready version is publicly available as the authoritative citation.