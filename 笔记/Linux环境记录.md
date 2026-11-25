# Ubuntu 环境记录（2026-11-23）

本机：Ubuntu 26.04.1 LTS / RTX 5060 Laptop 8GB / NVIDIA 595.99
仓库路径：`/home/zhichuan-xiao/机器人运控`（GitHub: `xuanyi520/Learn-Robot`）

## Python 环境

- Miniconda：`/home/zhichuan-xiao/miniconda3`
- 环境：`ai`（Python 3.12.16）
- 激活：`conda activate ai`
- 版本：torch 2.11.0+cu128（CUDA 可用）、mujoco 4.0.1、mink 1.4.0、gymnasium 1.4.0、numpy 2.5.4、scipy 1.18.1、matplotlib 3.11.2、pandas 3.1.1、ipykernel 7.4.0

## 运行讲义

A1 无窗口：

```bash
cd ~/机器人运控/讲义/A1_仿真循环
conda activate ai
python code.py --headless
```

A1 带窗口：`python code.py`

A2：

```bash
cd ~/机器人运控/讲义/A2_姿态与变换
conda activate ai
python code.py
```

## Notebook

- 内核已注册：`Python 3 (ai)`，路径 `~/.local/share/jupyter/kernels/ai`
- VS Code 已装 Python、Jupyter、中文语言包扩展
- 打开 `learn/Learn.ipynb`，内核选 `Python 3 (ai)`

## 与 Windows 的差异

- 本机没有 study MCP；查欠账、查讲义直接读仓库文件
- Windows 的 `E:\机器人运控` 对应 Linux 的 `/home/zhichuan-xiao/机器人运控`
- `代码/_fix_pycharm_conda_sdk.py`、`代码/_check_pycharm_sdk.py` 只适用于 Windows，Linux 上用不到

## 已验证

- A1 无窗口：4000 步跑完，最终角度 -180.00 度（与目标 180 度等价，相差 360 度），角速度约 0
- A2：四个姿态实验全部跑通，slerp 与线性插值对比正常
