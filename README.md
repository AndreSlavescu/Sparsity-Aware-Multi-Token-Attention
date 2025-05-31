# Sparsity-Aware-Multi-Token-Attention

## Create Environment
```bash
conda create -n smta python=3.10 -y
conda activate smta
```


### Build Liger Kernel

```bash
cd Liger-Kernel
pip install -e .
```

### Build Flame

```bash
cd flame
pip install .
```

#### Build Latest Version of FLA

```bash
pip uninstall flash-linear-attention && pip install -U --no-use-pep517 git+https://github.com/fla-org/flash-linear-attention
```

#### Build Torch Titan

install this specific version
```bash
pip install git+https://github.com/pytorch/torchtitan.git@5e2033c
```

### Running Experiments

```bash
chmod +x run_train_experiments.sh
./run_train_experiments.sh
```