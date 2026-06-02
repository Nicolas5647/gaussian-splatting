import subprocess
import sys
import argparse
import re
import os
import shutil
import csv
import itertools
import time

parser = argparse.ArgumentParser()
parser.add_argument("-s", type=str, required=True, help="Le chemin vers le dossier contenant les data")
parser.add_argument("--use_mask", action="store_true")

parser.add_argument("-r", nargs='+', default=["1"], help="Liste des résolutions")
parser.add_argument("--densify_grad_threshold", nargs='+', default=["0.0002"], help="Liste des thresholds de densification")
parser.add_argument("--scaling_lr", nargs='+', help="Liste des learning rates pour le scaling (optionnel)")
parser.add_argument("--densify_until_iter", nargs='+', help="Liste des itérations de densification s'arrêtant (optionnel)")

args, extra_args = parser.parse_known_args()

script_a_lancer = "train.py"
chemins_extraits = []
csv_file = "/app/output/training_metrics.csv"

mapping_suffix = {
    "-r": "r",
    "--densify_grad_threshold": "d",
    "--scaling_lr": "s",
    "--densify_until_iter": "ds"
}

mapping_header = {
    "-r": "Resolution",
    "--densify_grad_threshold": "Densification",
    "--scaling_lr": "Scaling",
    "--densify_until_iter": "Densification_Stop"
}

params_dict = {
    "-r": args.r,
    "--densify_grad_threshold": args.densify_grad_threshold,
}

if args.scaling_lr:
    params_dict["--scaling_lr"] = args.scaling_lr
if args.densify_until_iter:
    params_dict["--densify_until_iter"] = args.densify_until_iter

keys = list(params_dict.keys())
values = list(params_dict.values())
combinations = list(itertools.product(*values))

base_name = os.path.basename(args.s.strip(os.sep))

with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = ["Name"] + [mapping_header[k] for k in keys] + ["Mask", "Time_7000", "Time_30000"]
    writer.writerow(header)


for combo in combinations:
    current_params = dict(zip(keys, combo))

    # Génération du suffixe selon vos règles (r, d, s, ds)
    parts = []
    for k, v in current_params.items():
        short_key = mapping_suffix[k]
        parts.append(f"{short_key}{v}")
    
    suffixe = "_".join(parts)
    
    if args.use_mask:
        suffixe += "_mask"
        
    new_name = f"{base_name}_{suffixe}"
    model_path = os.path.join("output", new_name)
    
    commande = [sys.executable, "-u", script_a_lancer, "-s", args.s, "-m", model_path]
    
    # Ajouter des paramètres 
    for k, v in current_params.items():
        commande.extend([k, str(v)])
    
    if args.use_mask:
        commande.extend(["--use_mask", "True", "--random_background"])
    
    commande.extend(extra_args)

    # Affichage console
    params_str = " | ".join([f"{mapping_header[k]}: {v}" for k, v in current_params.items()])
    print(f"\n--- LANCEMENT : {params_str} ---")

    process = subprocess.Popen(
        commande, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1
    )

    time_7k = ""
    time_30k = ""

    for line in process.stdout:
        clean_line = line.strip('\r\n')
        
        if "Training progress" in clean_line:
            sys.stdout.write(f"\r{clean_line}")
            sys.stdout.flush()

            match_time = re.search(r"(\d+)/30000\s+\[([\d:]+)<", clean_line)
            if match_time:
                iteration = int(match_time.group(1))
                elapsed_time = match_time.group(2)
                if 6980 <= iteration <= 7050 and not time_7k:
                    time_7k = elapsed_time
                if iteration >= 29990:
                    time_30k = elapsed_time
        else:
            print(f"{clean_line}")
    
    process.wait()

    if process.returncode == 0:
        # Enregistrement dans le CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            row = [base_name] + list(combo) + [args.use_mask, time_7k, time_30k]
            writer.writerow(row)

        chemins_extraits.append(model_path)
    else:
        print(f"\nERREUR : Code {process.returncode}")

print("\nTerminé. Dossiers créés :")
for c in chemins_extraits:
    print(f"- {c}")
