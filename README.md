# cg2all-pyg

> **Fork of [huhlim/cg2all](https://github.com/huhlim/cg2all)** — DGL-free port on top of PyTorch Geometric + torch ≥ 2.2.
> The SE(3)-Transformer backbone has been re-implemented in-tree (`cg2all/lib/se3/`) so that **all 15 published checkpoints on [Zenodo 8393343](https://zenodo.org/record/8393343) load unchanged** (heavy-atom deviation < 2e-3 Å vs the DGL reference across all 10 CG models).
> Original project, paper, and checkpoints © Heo & Feig, licensed under Apache-2.0 (see `LICENSE`).
> Paper: **One bead per residue can describe all-atom protein structures**, *Structure* 32(1):97–111.e6 (2024) — [doi:10.1016/j.str.2023.10.013](https://doi.org/10.1016/j.str.2023.10.013) · [PMC10872525](https://pmc.ncbi.nlm.nih.gov/articles/PMC10872525/).

Convert coarse-grained protein structure to all-atom model.

## Legacy (DGL) vs this fork (PyG) — benchmark

CPU, 3 repeats per case, same Zenodo 8393343 checkpoints loaded in both envs.

| case | input | legacy fwd (s) | this fork fwd (s) | speed | heavy-atom max Δ (Å) |
|---|---|---:|---:|---:|---:|
| 1ab1 Cα        | `1ab1_A.calpha.pdb`               | 1.024 | 1.455 | 0.70× | **0.000000** (bit-exact) |
| 1ab1 Residue   | `1ab1_A.residue.pdb`              | 0.706 | 0.826 | 0.85× | 0.001 |
| 1ab1 Backbone  | `1ab1_A.bb.pdb`                   | 0.513 | 0.819 | 0.63× | 0.001 |
| 1ab1 Martini   | `1ab1_A.martini.pdb`              | 1.992 | 0.621 | **3.21×** | 0.001 |
| 1jni Cα + DCD  | `1jni.calpha.{pdb,dcd}` (5 frames)| 4.699 | 2.310 | **2.03×** | 6.3 × 10⁻⁴ |

**Numerics**: every case matches legacy within the 1-ULP PDB coordinate resolution (≤ 1e-3 Å), with 1ab1 Cα bit-exact. Across all 10 CG models the heavy-atom deviation stays below 2e-3 Å.

**Speed**: on small single-PDB cases the DGL CPU C++ kernels still beat native `scatter_reduce` by ~25–40 %; on larger graphs (Martini) and repeated forward passes (DCD trajectory) the torch 2.5 path wins by **2–3×**. GPU results pending.

## Upstream-hosted demos (run the original DGL version)

These public demos are hosted by the upstream authors and run the **DGL-based `huhlim/cg2all` codebase**, not this fork. They are linked here for convenience; this fork does not ship hosted demos.

- [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/huhlim/cg2all) — interactive CG → all-atom conversion.
- [![cg2all notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huhlim/cg2all/blob/main/cg2all.ipynb) — `convert_all2cg`, `convert_cg2all`, and CG-trajectory → atomistic-trajectory workflows in Google Colab.
- [![cryo-EM notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huhlim/cg2all/blob/main/cryo_em_minimizer.ipynb) — cryo-EM density-map local optimization with `cryo_em_minimizer` in Google Colab.

## Installation
This package depends on PyTorch (≥ 2.2), PyTorch Geometric, [a modified MDTraj](https://github.com/huhlim/mdtraj), and standard scientific Python libraries. It no longer depends on DGL or the legacy `huhlim/SE3Transformer` fork — the SE(3)-Transformer has been ported in-tree (`cg2all/lib/se3/`) and the existing Zenodo 8393343 checkpoints load unchanged. Installing this package also places the executables `convert_cg2all`, `convert_all2cg`, and `cryo_em_minimizer` on your PATH.

Tested on Linux.

#### CPU
```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch_cluster -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
python -m pip install git+https://github.com/ding-oh/cg2all-pyg
```

#### CUDA (GPU)
Replace the wheel index with the CUDA variant matching your driver/toolkit. Example for CUDA 12.4:
```bash
conda create -n cg2all-pyg python=3.11 -y && conda activate cg2all-pyg
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch_cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
python -m pip install git+https://github.com/ding-oh/cg2all-pyg
```
For other CUDA versions, pick the matching index at <https://download.pytorch.org/whl/> and <https://data.pyg.org/whl/>.

#### cryo_em_minimizer
`mrcfile` is included in the package dependencies and installed automatically.

## Usages
### convert_cg2all
convert a coarse-grained protein structure to all-atom model
```bash
usage: convert_cg2all [-h] -p IN_PDB_FN [-d IN_DCD_FN] -o OUT_FN [-opdb OUTPDB_FN]
                      [--cg {supported_cg_models}] [--chain-break-cutoff CHAIN_BREAK_CUTOFF] [-a]
                      [--fix] [--ckpt CKPT_FN] [--time TIME_JSON] [--device DEVICE] [--batch BATCH_SIZE] [--proc N_PROC]

options:
  -h, --help            show this help message and exit
  -p IN_PDB_FN, --pdb IN_PDB_FN
  -d IN_DCD_FN, --dcd IN_DCD_FN
  -o OUT_FN, --out OUT_FN, --output OUT_FN
  -opdb OUTPDB_FN
  --cg {supported_cg_models}
  --chain-break-cutoff CHAIN_BREAK_CUTOFF
  -a, --all, --is_all
  --fix, --fix_atom
  --standard-name
  --ckpt CKPT_FN
  --time TIME_JSON
  --device DEVICE
  --batch BATCH_SIZE
  --proc N_PROC
```
#### arguments
* -p/--pdb: Input PDB file (**mandatory**).
* -d/--dcd: Input DCD file (optional). If a DCD file is given, the input PDB file will be used to define its topology.
* -o/--out/--output: Output PDB or DCD file (**mandatory**). If a DCD file is given, it will be a DCD file. Otherwise, a PDB file will be created.
* -opdb: If a DCD file is given, it will write the last snapshot as a PDB file. (optional)
* --cg: Coarse-grained representation to use (optional, default=CalphaBasedModel).
  - CalphaBasedModel: CA-trace (atom names should be "CA")
  - ResidueBasedModel: Residue center-of-mass (atom names should be "CA")
  - SidechainModel: Sidechain center-of-mass (atom names should be "SC")
  - CalphaCMModel: CA-trace + Residue center-of-mass (atom names should be "CA" and "CM")
  - CalphaSCModel: CA-trace + Sidechain center-of-mass (atom names should be "CA" and "SC")
  - BackboneModel: Model only with backbone atoms (N, CA, C)
  - MainchainModel: Model only with mainchain atoms (N, CA, C, O)
  - Martini: [Martini](http://cgmartini.nl/) model
  - Martini3: [Martini3](http://www.cgmartini.nl/index.php/martini-3-0) model
  - PRIMO: [PRIMO](http://dx.doi.org/10.1002/prot.22645) model
* --chain-break-cutoff: The CA-CA distance cutoff that determines chain breaks. (default=10 Angstroms)
* --fix/--fix_atom: preserve coordinates in the input CG model. For example, CA coordinates in a CA-trace model will be kept in its cg2all output model.
* --standard-name: output atom names follow the IUPAC nomenclature. (default=False; output atom names will use CHARMM atom names)
* --ckpt: Input PyTorch ckpt file (optional). If a ckpt file is given, it will override "--cg" option.
* --time: Output JSON file for recording timing. (optional)
* --device: Specify a device to run the model. (optional) You can choose "cpu" or "cuda", or the script will detect one automatically. </br>
  "**cpu**" is usually faster than "cuda" unless the input/output system is really big or you provided a DCD file with many frames because it takes a lot for loading a model ckpt file on a GPU.
* --batch: the number of frames to be dealt at a time. (optional, default=1)
* --proc: Specify the number of threads for loading input data. It is only used for dealing with a DCD file. (optional, default=OMP_NUM_THREADS or 1)

#### examples
Conversion of a PDB file
```bash
convert_cg2all -p tests/1ab1_A.calpha.pdb -o tests/1ab1_A.calpha.all.pdb --cg CalphaBasedModel
```
Conversion of a DCD trajectory file
```bash
convert_cg2all -p tests/1jni.calpha.pdb -d tests/1jni.calpha.dcd -o tests/1jni.calpha.all.dcd --cg CalphaBasedModel
```
Conversion of a PDB file using a ckpt file
```bash
convert_cg2all -p tests/1ab1_A.calpha.pdb -o tests/1ab1_A.calpha.all.pdb --ckpt CalphaBasedModel-104.ckpt
```
<hr/>

### convert_all2cg
convert an all-atom protein structure to coarse-grained model
```bash
usage: convert_all2cg [-h] -p IN_PDB_FN [-d IN_DCD_FN] -o OUT_FN [--cg {supported_cg_models}]

options:
  -h, --help            show this help message and exit
  -p IN_PDB_FN, --pdb IN_PDB_FN
  -d IN_DCD_FN, --dcd IN_DCD_FN
  -o OUT_FN, --out OUT_FN, --output OUT_FN
  --cg
```
#### arguments
* -p/--pdb: Input PDB file (**mandatory**).
* -d/--dcd: Input DCD file (optional). If a DCD file is given, the input PDB file will be used to define its topology.
* -o/--out/--output: Output PDB or DCD file (**mandatory**). If a DCD file is given, it will be a DCD file. Otherwise, a PDB file will be created.
* --cg: Coarse-grained representation to use (optional, default=CalphaBasedModel).
  - CalphaBasedModel: CA-trace (atom names should be "CA")
  - ResidueBasedModel: Residue center-of-mass (atom names should be "CA")
  - SidechainModel: Sidechain center-of-mass (atom names should be "SC")
  - CalphaCMModel: CA-trace + Residue center-of-mass (atom names should be "CA" and "CM")
  - CalphaSCModel: CA-trace + Sidechain center-of-mass (atom names should be "CA" and "SC")
  - BackboneModel: Model only with backbone atoms (N, CA, C)
  - MainchainModel: Model only with mainchain atoms (N, CA, C, O)
  - Martini: [Martini](http://cgmartini.nl/) model
  - Martini3: [Martini3](http://www.cgmartini.nl/index.php/martini-3-0) model
  - PRIMO: [PRIMO](http://dx.doi.org/10.1002/prot.22645) model
  
#### an example
```bash
convert_all2cg -p tests/1ab1_A.pdb -o tests/1ab1_A.calpha.pdb --cg CalphaBasedModel
```

<hr/>

### script/cryo_em_minimizer.py 
Local optimization of protein model structure against given electron density map. This script is a proof-of-concept that utilizes cg2all network to optimize at CA-level resolution with objective functions in both atomistic and CA-level resolutions. It is highly recommended to use **cuda** environment.
```bash
usage: cryo_em_minimizer [-h] -p IN_PDB_FN -m IN_MAP_FN -o OUT_DIR [-a]
                         [-n N_STEP] [--freq OUTPUT_FREQ]
                         [--chain-break-cutoff CHAIN_BREAK_CUTOFF]
                         [--restraint RESTRAINT]
                         [--cg {CalphaBasedModel,CA,ca,ResidueBasedModel,RES,res}]
                         [--standard-name] [--uniform_restraint]
                         [--nonuniform_restraint] [--segment SEGMENT_S]

options:
  -h, --help            show this help message and exit
  -p IN_PDB_FN, --pdb IN_PDB_FN
  -m IN_MAP_FN, --map IN_MAP_FN
  -o OUT_DIR, --out OUT_DIR, --output OUT_DIR
  -a, --all, --is_all
  -n N_STEP, --step N_STEP
  --freq OUTPUT_FREQ, --output_freq OUTPUT_FREQ
  --chain-break-cutoff CHAIN_BREAK_CUTOFF
  --restraint RESTRAINT
  --cg {CalphaBasedModel,CA,ca,ResidueBasedModel,RES,res}
  --standard-name
  --uniform_restraint
  --nonuniform_restraint
  --segment SEGMENT_S
```
#### arguments
* -p/--pdb: Input PDB file (**mandatory**).
* -m/--map: Input electron density map file in the MRC or CCP4 format (**mandatory**).
* -o/--out/--output: Output directory to save optimized structures (**mandatory**).
* -a/--all/--is_all: Whether the input PDB file is atomistic structure or not. (optional, default=False)
* -n/--step: The number of minimization steps. (optional, default=1000)
* --freq/--output_freq: The interval between saving intermediate outputs. (optional, default=100)
* --chain-break-cutoff: The CA-CA distance cutoff that determines chain breaks. (default=10 Angstroms)
* --restraint: The weight of distance restraints. (optional, default=100.0)
* --cg: Coarse-grained representation to use (default=ResidueBasedModel)
* --standard-name: output atom names follow the IUPAC nomenclature. (default=False; output atom names will use CHARMM atom names)
*  --uniform_restraint/--nonuniform_restraint: Whether to use uniform restraints. (default=True) If it is set to False,
    the restraint weights will be dependent on the pLDDT values recorded in the PDB file's B-factor columns. 
*  --segment: The segmentation method for applying rigid-body operations. (default=None)
    * None: Input structure is not segmented, so the same rigid-body operations are applied to the whole structure.
    * chain: Input structure is segmented based on chain IDs. Rigid-body operations are independently applied to each chain.
    * segment: Similar to "chain" option, but the structure is segmented based on peptide bond connectivities. 
    * 0-99,100-199: Explicit segmentation based on the 0-index based residue numbers.

#### an example
```bash
./cg2all/script/cryo_em_minimizer.py -p tests/3isr.af2.pdb -m tests/3isr_5.mrc -o 3isr_5+3isr.af2 --all
```

## Datasets
The training/validation/test sets are available at [zenodo](https://zenodo.org/record/8273739).


## Citation

This fork only re-implements the runtime; **all scientific credit — model design, training, checkpoints — belongs to the original authors**. If you use this package, please cite the original paper:

> Lim Heo & Michael Feig, "One bead per residue can describe all-atom protein structures", *Structure* **32**(1):97–111.e6 (**2024**). [doi:10.1016/j.str.2023.10.013](https://doi.org/10.1016/j.str.2023.10.013) · [PMC10872525](https://pmc.ncbi.nlm.nih.gov/articles/PMC10872525/) · preprint (earlier title: "One particle per residue is sufficient…"): [bioRxiv 2023.05.22.541652](https://www.biorxiv.org/content/10.1101/2023.05.22.541652v1)

BibTeX (published version):
```bibtex
@article{heo2024cg2all,
  title   = {One bead per residue can describe all-atom protein structures},
  author  = {Heo, Lim and Feig, Michael},
  journal = {Structure},
  volume  = {32},
  number  = {1},
  pages   = {97--111.e6},
  year    = {2024},
  doi     = {10.1016/j.str.2023.10.013},
  pmid    = {38000367},
  pmcid   = {PMC10872525}
}
```

Upstream project: [huhlim/cg2all](https://github.com/huhlim/cg2all) — [Zenodo DOI](https://zenodo.org/doi/10.5281/zenodo.10009208)
