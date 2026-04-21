"""Regression tests for `convert_cg2all`.

Each test loads a committed golden output under `tests/regression/golden/` and
runs `convert_cg2all` on the matching input sample. Heavy-atom coordinates are
compared via mdtraj so the binary DCD trajectory case gets the same treatment
as the text-PDB cases.

Run:
    pytest tests/regression/ -v
"""
from __future__ import annotations

import pathlib
import subprocess
import tempfile

import mdtraj as md
import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"
GOLDEN_DIR = pathlib.Path(__file__).with_name("golden")

# PDB coordinates are stored at 3 decimal A, so 1 ULP == 1e-3 A. We accept up
# to 2 ULP on PDB and a tighter 5e-3 A on DCD (which is binary float32, no
# rounding).
PDB_ATOM_TOL_A = 2e-3
DCD_ATOM_TOL_A = 5e-3

CG_TO_SAMPLE = {
    "CalphaBasedModel":  "1ab1_A.calpha.pdb",
    "ResidueBasedModel": "1ab1_A.residue.pdb",
    "SidechainModel":    "1ab1_A.sc.pdb",
    "CalphaCMModel":     "1ab1_A.cacm.pdb",
    "CalphaSCModel":     "1ab1_A.casc.pdb",
    "BackboneModel":     "1ab1_A.bb.pdb",
    "MainchainModel":    "1ab1_A.mc.pdb",
    "Martini":           "1ab1_A.martini.pdb",
    "Martini3":          "1ab1_A.martini3.pdb",
    "PRIMO":             "1ab1_A.primo.pdb",
}

DCD_CASES = [
    # (case_id, input_pdb, input_dcd, cg_model)
    ("1jni_CA_DCD", "1jni.calpha.pdb", "1jni.calpha.dcd", "CalphaBasedModel"),
]


def _heavy_xyz_A(traj: md.Trajectory) -> np.ndarray:
    """Return an (n_frames, n_heavy_atoms, 3) array of coords in Angstrom.

    mdtraj stores internal coords in nm, so we convert with x10.  We intentionally
    select heavy atoms by element to avoid relying on hydrogen naming conventions
    that differ across the CHARMM/IUPAC output modes.
    """
    heavy = traj.top.select("element != H")
    return traj.xyz[:, heavy, :] * 10.0


def _max_heavy_dev(new_path: pathlib.Path, golden_path: pathlib.Path,
                   topology: md.Topology | None = None) -> float:
    """Max |ΔR| over all heavy atoms (and all frames for DCD)."""
    if topology is None:
        new_traj = md.load(str(new_path))
        golden_traj = md.load(str(golden_path))
    else:
        new_traj = md.load(str(new_path), top=topology)
        golden_traj = md.load(str(golden_path), top=topology)
    new_xyz = _heavy_xyz_A(new_traj)
    golden_xyz = _heavy_xyz_A(golden_traj)
    if new_xyz.shape != golden_xyz.shape:
        raise AssertionError(
            f"shape mismatch: new={new_xyz.shape} golden={golden_xyz.shape}"
        )
    return float(np.abs(new_xyz - golden_xyz).max())


@pytest.mark.parametrize("cg,sample_name", list(CG_TO_SAMPLE.items()))
def test_convert_cg2all_pdb(cg: str, sample_name: str) -> None:
    sample = TESTS_DIR / sample_name
    golden = GOLDEN_DIR / f"{cg}.pdb"
    if not sample.exists():
        pytest.skip(f"sample input missing: {sample}")
    if not golden.exists():
        pytest.skip(f"golden reference missing: {golden}")

    with tempfile.TemporaryDirectory() as tmp:
        out_path = pathlib.Path(tmp) / f"{cg}.pdb"
        subprocess.run(
            [
                "convert_cg2all",
                "-p", str(sample),
                "-o", str(out_path),
                "--cg", cg,
                "--device", "cpu",
            ],
            check=True,
        )
        max_dev = _max_heavy_dev(out_path, golden)

    assert max_dev < PDB_ATOM_TOL_A, (
        f"[{cg}] max heavy-atom deviation {max_dev:.6f} Å "
        f"exceeds PDB tolerance {PDB_ATOM_TOL_A} Å"
    )


@pytest.mark.parametrize("case_id,pdb_name,dcd_name,cg", DCD_CASES)
def test_convert_cg2all_dcd(case_id: str, pdb_name: str, dcd_name: str, cg: str) -> None:
    """DCD trajectory regression — compares all frames, heavy atoms only."""
    sample_pdb = TESTS_DIR / pdb_name
    sample_dcd = TESTS_DIR / dcd_name
    golden_dcd = GOLDEN_DIR / f"{case_id}.dcd"
    golden_topo = GOLDEN_DIR / f"{case_id}.topo.pdb"
    for p in (sample_pdb, sample_dcd):
        if not p.exists():
            pytest.skip(f"sample input missing: {p}")
    for p in (golden_dcd, golden_topo):
        if not p.exists():
            pytest.skip(f"golden reference missing: {p}")

    topology = md.load(str(golden_topo)).topology

    with tempfile.TemporaryDirectory() as tmp:
        out_dcd = pathlib.Path(tmp) / f"{case_id}.dcd"
        subprocess.run(
            [
                "convert_cg2all",
                "-p", str(sample_pdb),
                "-d", str(sample_dcd),
                "-o", str(out_dcd),
                "--cg", cg,
                "--device", "cpu",
            ],
            check=True,
        )
        max_dev = _max_heavy_dev(out_dcd, golden_dcd, topology=topology)

    assert max_dev < DCD_ATOM_TOL_A, (
        f"[{case_id}] max heavy-atom deviation {max_dev:.6f} Å "
        f"exceeds DCD tolerance {DCD_ATOM_TOL_A} Å"
    )
