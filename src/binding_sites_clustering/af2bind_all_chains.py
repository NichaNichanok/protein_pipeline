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

def get_all_chains(pdb_filename):
    """
    Extract all chain IDs from the PDB file.

    Args:
        pdb_filename (str): Path to the PDB file.

    Returns:
        list: List of unique chain IDs in the PDB file.
    """
    chains = set()
    with open(pdb_filename, 'r') as file:
        for line in file:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                chains.add(line[21].strip())
    return sorted(list(chains))


def run_af2bind_all_chains(target_pdb, target_chains=None, mask_sidechains=True, mask_sequence=False):
    """
    Run AF2Bind for all chains in a protein by default or selected chains.

    Args:
        target_pdb (str): Path to the PDB file or PDB code.
        target_chains (list of str, optional): List of chain IDs to analyze. If None, process all chains.
        mask_sidechains (bool): Mask sidechains in the target (default: True).
        mask_sequence (bool): Mask sequence in the target (default: False).

    Returns:
        pd.DataFrame: Combined DataFrame of results for all processed chains.
    """
    target_pdb = target_pdb.replace(" ", "")
    pdb_filename = get_pdb(target_pdb)

    # Get all chains if not provided
    if target_chains is None:
        target_chains = get_all_chains(pdb_filename)
        print(f"Processing all chains: {target_chains}")
    else:
        print(f"Processing specified chains: {target_chains}")

    clear_mem()
    af_model = mk_afdesign_model(protocol="binder", debug=True)

    all_results = []  # List to store results for all chains

    for chain_id in target_chains:
        chain_id = chain_id.strip()
        print(f"Processing chain: {chain_id}")

        # Prepare inputs for the current chain
        af_model.prep_inputs(pdb_filename=pdb_filename,
                             chain=chain_id,
                             binder_len=20,
                             rm_target_sc=mask_sidechains,
                             rm_target_seq=mask_sequence)

        # Split
        r_idx = af_model._inputs["residue_index"][-20] + (1 + np.arange(20)) * 50
        af_model._inputs["residue_index"][-20:] = r_idx.flatten()

        af_model.set_seq("ACDEFGHIKLMNPQRSTVWY")
        af_model.predict(verbose=False)

        o = af2bind(af_model.aux["debug"]["outputs"], mask_sidechains=mask_sidechains)
        pred_bind = o["p_bind"].copy()
        pred_bind_aa = o["p_bind_aa"].copy()

        # Process results for the current chain
        labels = ["chain", "resi", "resn", "p(bind)"]
        data = []
        for i in range(af_model._target_len):
            c = af_model._pdb["idx"]["chain"][i]
            r = af_model._pdb["idx"]["residue"][i]
            a = aa_order.get(af_model._pdb["batch"]["aatype"][i], "X")
            p = pred_bind[i]
            data.append([c, r, a, p])

        df = pd.DataFrame(data, columns=labels)
        df.to_csv(f'results_{target_pdb}_chain_{chain_id}.csv')
        all_results.append(df)

        # Print top binding residues for the current chain
        df_sorted = df.sort_values("p(bind)", ascending=False, ignore_index=True).rename_axis('rank').reset_index()
        print(df_sorted.head(15))

    # Combine all results into one DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(f'results_{target_pdb}_all_chains.csv')
    print(f"Combined results saved to: results_{target_pdb}_all_chains.csv")
    return combined_df


def main():
    parser = argparse.ArgumentParser(description="Run AlphaFold2 and Binding Analysis")
    parser.add_argument("target", metavar="TARGET", type=str, help="Protein structure file or PDB code")
    parser.add_argument("-c", "--chains", type=str, nargs='+', default=None,
                        help="List of target chains. If not provided, process all chains.")
    parser.add_argument("-s", "--mask_sidechains", action="store_true", help="Mask sidechains (default: False)")
    parser.add_argument("-m", "--mask_sequence", action="store_true", help="Mask sequence (default: False)")
    args = parser.parse_args()

    run_af2bind_all_chains(target_pdb=args.target, target_chains=args.chains,
                           mask_sidechains=args.mask_sidechains, mask_sequence=args.mask_sequence)


if __name__ == "__main__":
    main()
    ## To run this script:
    ## Example 1 (all chains): !python af2bind_local.py 7CEI -s -m
    ## Example 2 (specific chains): !python af2bind_local.py 7CEI -c A B -s -m
 