### This code is translated to run locally from this website: https://github.com/sokrypton/af2bind
### For the commandline used
import argparse
import copy
import os
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import py3Dmol
from colabdesign import clear_mem, mk_afdesign_model
from colabdesign.af.alphafold.common import protein, residue_constants
from scipy.special import expit as sigmoid

#export AF2_MODEL_DIR=$(pwd)/params

# Define aa_order dictionary
aa_order = {v: k for k, v in residue_constants.restype_order.items()}

def get_pdb(pdb_code=""):
    if pdb_code is None or pdb_code == "":
        pdb_file = input("Please provide the path to your PDB file: ")
        return pdb_file
    elif os.path.isfile(pdb_code):
        return pdb_code
    elif len(pdb_code) == 4:
        os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
        print(f"Dowloaded the {pdb_code} from rcsb-website")
        return f"{pdb_code}.pdb"
    else:
        os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v4.pdb")
        return f"AF-{pdb_code}-F1-model_v4.pdb"

def af2bind(outputs, mask_sidechains=True, seed=0):
    pair_A = outputs["representations"]["pair"][:-20, -20:]
    pair_B = outputs["representations"]["pair"][-20:, :-20].swapaxes(0, 1)
    pair_A = pair_A.reshape(pair_A.shape[0], -1)
    pair_B = pair_B.reshape(pair_B.shape[0], -1)
    x = np.concatenate([pair_A, pair_B], -1)

    # Get params
    if mask_sidechains:
        model_type = f"split_nosc_pair_A_split_nosc_pair_B_{seed}"
    else:
        model_type = f"split_pair_A_split_pair_B_{seed}"
    with open(f"af2bind_params/attempt_7_2k_lam0-03/{model_type}.pickle", "rb") as handle:
        params_ = pickle.load(handle)
    params_ = dict(**params_["~"], **params_["linear"])
    p = jax.tree_map(lambda x: np.asarray(x), params_)

    # Get predictions
    x = (x - p["mean"]) / p["std"]
    x = (x * p["w"][:, 0]) + (p["b"] / x.shape[-1])
    p_bind_aa = x.reshape(x.shape[0], 2, 20, -1).sum((1, 3))
    p_bind = sigmoid(p_bind_aa.sum(-1))
    return {"p_bind": p_bind, "p_bind_aa": p_bind_aa}

def run_af2bind(target_pdb, target_chain, mask_sidechains=True, mask_sequence=False):
    target_pdb = target_pdb.replace(" ", "")
    target_chain = target_chain.replace(" ", "")
    if target_chain == "":
        target_chain = "A"

    pdb_filename = get_pdb(target_pdb)

    clear_mem()
    af_model = mk_afdesign_model(protocol="binder", debug=True)
    af_model.prep_inputs(pdb_filename=pdb_filename,
                         chain=target_chain,
                         binder_len=20,
                         rm_target_sc=mask_sidechains,
                         rm_target_seq=mask_sequence)

    # Split
    r_idx = af_model._inputs["residue_index"][-20] + (1 + np.arange(20)) * 50
    af_model._inputs["residue_index"][-20:] = r_idx.flatten()

    af_model.set_seq("ACDEFGHIKLMNPQRSTVWY")
    af_model.predict(verbose=False)

    o = af2bind(af_model.aux["debug"]["outputs"],
                mask_sidechains=mask_sidechains)
    pred_bind = o["p_bind"].copy()
    pred_bind_aa = o["p_bind_aa"].copy()

    # Process results
    labels = ["chain", "resi", "resn", "p(bind)"]
    data = []
    for i in range(af_model._target_len):
        c = af_model._pdb["idx"]["chain"][i]
        r = af_model._pdb["idx"]["residue"][i]
        a = aa_order.get(af_model._pdb["batch"]["aatype"][i], "X")
        p = pred_bind[i]
        data.append([c, r, a, p])

    df = pd.DataFrame(data, columns=labels)
    df.to_csv(f'results_{target_pdb}.csv')

    df_sorted = df.sort_values("p(bind)", ascending=False, ignore_index=True).rename_axis('rank').reset_index()
    print(df_sorted.head(15))

    top_n = 15
    top_n_idx = pred_bind.argsort()[::-1][:15]
    pymol_cmd = "select ch" + str(target_chain) + ","
    for n, i in enumerate(top_n_idx):
        p = pred_bind[i]
        c = af_model._pdb["idx"]["chain"][i]
        r = af_model._pdb["idx"]["residue"][i]
        pymol_cmd += f" resi {r}"
        if n < top_n - 1:
            pymol_cmd += " +"

    print("\n🧪 Pymol Selection Cmd:")
    print(pymol_cmd)

def main():
    parser = argparse.ArgumentParser(description="Run AlphaFold2 and Binding Analysis")
    parser.add_argument("target", metavar="TARGET", type=str, help="Protein structure file or PDB code")
    parser.add_argument("-c", "--chain", type=str, default="", help="Target chain (default: A)")
    parser.add_argument("-s", "--mask_sidechains", action="store_true", help="Mask sidechains (default: False)")
    parser.add_argument("-m", "--mask_sequence", action="store_true", help="Mask sequence (default: False)")
    args = parser.parse_args()

    run_af2bind(target_pdb=args.target, target_chain=args.chain, mask_sidechains=args.mask_sidechains, mask_sequence=args.mask_sequence)

if __name__ == "__main__":
    main()
    ## To run this script:
    ## !python src/binding_sites_clustering/af2bind_local.py 7CEI -c A -s -m
