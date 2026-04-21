"""Python-level API for `convert_cg2all` and `convert_all2cg` — DGL-free port.

Mirrors the signatures of the legacy `cg2all.lib.snippets` module so the
Jupyter notebooks (`cg2all.ipynb`, `cryo_em_minimizer.ipynb`) continue to work
with `from cg2all import convert_cg2all, convert_all2cg`.
"""
from __future__ import annotations

import functools
import os
import pathlib
import warnings
from typing import Optional

import mdtraj
import numpy as np
import torch
from torch.utils.data import DataLoader

from cg2all.lib import libcg, libmodel
from cg2all.lib.graph import batch_graphs
from cg2all.lib.libconfig import MODEL_HOME
from cg2all.lib.libdata import (
    PredictionData,
    create_topology_from_data,
    create_trajectory_from_batch,
)
from cg2all.lib.libpdb import write_SSBOND
from cg2all.lib.libter import patch_termini
from cg2all.lib.residue_constants import read_coarse_grained_topology

warnings.filterwarnings("ignore")


_CG_MODEL_MAP = {
    "CalphaBasedModel":  libcg.CalphaBasedModel,
    "ResidueBasedModel": libcg.ResidueBasedModel,
    "SidechainModel":    libcg.SidechainModel,
    "Martini":           libcg.Martini,
    "Martini3":          libcg.Martini3,
    "PRIMO":             libcg.PRIMO,
    "CalphaCMModel":     libcg.CalphaCMModel,
    "CalphaSCModel":     libcg.CalphaSCModel,
    "BackboneModel":     libcg.BackboneModel,
    "MainchainModel":    libcg.MainchainModel,
}


def convert_cg2all(
    in_pdb_fn,
    out_fn,
    model_type: str = "CalphaBasedModel",
    in_dcd_fn: Optional[str] = None,
    ckpt_fn: Optional[str] = None,
    fix_atom: bool = False,
    device: Optional[str] = None,
    n_proc: Optional[int] = None,
):
    """Reconstruct an all-atom structure from a coarse-grained model.

    Parameters mirror the legacy API. `model_type` selects the CG scheme
    (and therefore which pretrained checkpoint is loaded from `MODEL_HOME`);
    see keys of `_CG_MODEL_MAP`.
    """
    if n_proc is None:
        n_proc = int(os.getenv("OMP_NUM_THREADS", 1))

    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    if ckpt_fn is None:
        suffix = "-FIX.ckpt" if fix_atom else ".ckpt"
        ckpt_fn = MODEL_HOME / f"{model_type}{suffix}"
    ckpt = torch.load(ckpt_fn, map_location=device_t)
    config = ckpt["hyper_parameters"]

    if config["cg_model"] not in _CG_MODEL_MAP:
        raise KeyError(f"Unknown cg_model in checkpoint: {config['cg_model']}")
    cg_model = _CG_MODEL_MAP[config["cg_model"]]
    config = libmodel.set_model_config(config, cg_model, flattened=False)
    model = libmodel.Model(config, cg_model, compute_loss=False)

    state_dict = ckpt["state_dict"]
    for key in list(state_dict):
        state_dict[".".join(key.split(".")[1:])] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to(device_t)
    model.set_constant_tensors(device_t)
    model.eval()

    input_s = PredictionData(
        in_pdb_fn,
        cg_model,
        dcd_fn=in_dcd_fn,
        radius=config.globals.radius,
        fix_atom=config.globals.fix_atom,
    )
    unitcell_lengths = unitcell_angles = None
    if in_dcd_fn is not None:
        unitcell_lengths = input_s.cg.unitcell_lengths
        unitcell_angles = input_s.cg.unitcell_angles

    use_loader = len(input_s) > 1 and n_proc > 1
    if use_loader:
        loader = DataLoader(
            input_s, batch_size=1, num_workers=n_proc,
            shuffle=False, collate_fn=lambda xs: batch_graphs(xs),
        )
    else:
        loader = None

    if in_dcd_fn is None:
        batch = input_s[0].to(device_t)
        with torch.no_grad():
            R = model.forward(batch)[0]["R"]
        traj_s, ssbond_s = create_trajectory_from_batch(batch, R)
        output = patch_termini(traj_s[0])
        output.save(str(out_fn))
        if len(ssbond_s[0]) > 0:
            write_SSBOND(str(out_fn), output.top, ssbond_s[0])
        return output

    xyz = []
    iterable = loader if loader is not None else [input_s[i] for i in range(len(input_s))]
    for batch in iterable:
        batch = batch.to(device_t)
        with torch.no_grad():
            R = model.forward(batch)[0]["R"].cpu().detach().numpy()
            mask = batch.node["output_atom_mask"].cpu().detach().numpy()
            xyz.append(R[mask > 0.0])
    top, atom_index = create_topology_from_data(batch)
    xyz = np.array(xyz)[:, atom_index]
    traj = mdtraj.Trajectory(
        xyz=xyz,
        topology=top,
        unitcell_lengths=unitcell_lengths,
        unitcell_angles=unitcell_angles,
    )
    output = patch_termini(traj)
    output.save(str(out_fn))
    return output


def convert_all2cg(
    in_pdb_fn,
    out_fn,
    model_type: str = "CalphaBasedModel",
    in_dcd_fn: Optional[str] = None,
):
    """All-atom → CG, deterministic (no trained model)."""
    aliases = {
        "CA": "CalphaBasedModel", "ca": "CalphaBasedModel",
        "RES": "ResidueBasedModel", "res": "ResidueBasedModel",
        "martini": "Martini",
        "martini3": "Martini3",
        "primo": "PRIMO",
        "CACM": "CalphaCMModel", "cacm": "CalphaCMModel", "CalphaCM": "CalphaCMModel",
        "CASC": "CalphaSCModel", "casc": "CalphaSCModel", "CalphaSC": "CalphaSCModel",
        "SC": "SidechainModel", "sc": "SidechainModel", "sidechain": "SidechainModel",
        "BB": "BackboneModel", "bb": "BackboneModel", "backbone": "BackboneModel", "Backbone": "BackboneModel",
        "MC": "MainchainModel", "mc": "MainchainModel", "mainchain": "MainchainModel", "Mainchain": "MainchainModel",
    }
    canonical = aliases.get(model_type, model_type)
    if canonical not in _CG_MODEL_MAP:
        raise KeyError(f"Unknown CG model: {model_type}")
    cls = _CG_MODEL_MAP[canonical]

    # Martini/Martini3/PRIMO need a topology_map injected.
    if canonical in ("Martini", "Martini3", "PRIMO"):
        topology_map = read_coarse_grained_topology(
            {"Martini": "martini", "Martini3": "martini3", "PRIMO": "primo"}[canonical]
        )
        cls = functools.partial(cls, topology_map=topology_map)

    cg = cls(in_pdb_fn, dcd_fn=in_dcd_fn)
    if in_dcd_fn is None:
        cg.write_cg(cg.R_cg, pdb_fn=str(out_fn))
        if len(cg.ssbond_s) > 0:
            write_SSBOND(str(out_fn), cg.top, cg.ssbond_s)
    else:
        cg.write_cg(cg.R_cg, dcd_fn=str(out_fn))
