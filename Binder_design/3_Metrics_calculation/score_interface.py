#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本脚本用于批量计算PDB文件的界面分数 (带即时耗时)。

它会扫描一个输入目录 (默认为 ./pdbs_scores) 中所有匹配 *_model.pdb 的文件，
计算每个文件的界面分数，并将所有结果汇总到一个CSV文件中。
在 Slurm 中运行时，它会为每个文件打印处理耗时。

运行示例:
python calculate_scores_batch_timing.py \
    --input_dir ./pdbs_scores \
    --output_csv ./all_scores.csv \
    --binder_chain A \
    --dalphaball_path /path/to/your/bindcraft/functions/DAlphaBall.gcc
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import time  # <-- 1. 导入 time 模块
from pprint import pprint

try:
    # --- PyRosetta 核心依赖 ---
    import pyrosetta as pr
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
    from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
    from pyrosetta.rosetta.core.simple_metrics.metrics import TotalEnergyMetric, SasaMetric
    from pyrosetta.rosetta.core.select.residue_selector import LayerSelector

    # --- BioPython 核心依赖 ---
    from Bio.PDB import PDBParser, Selection
    from scipy.spatial import cKDTree
    from Bio.PDB.Selection import unfold_entities

except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已正确安装 PyRosetta, BioPython, NumPy, SciPy 和 Pandas。")
    print("pip install pandas")
    print("PyRosetta 需要单独安装和许可证。")
    exit(1)


####################################################################
# 依赖函数 (来自 biopython_utils.py)
# [函数 'hotspot_residues' 和 'three_to_one_map' 在此省略，与上一版相同]
####################################################################

# hotspot_residues 函数需要这个字典
three_to_one_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def hotspot_residues(trajectory_pdb, binder_chain="A", atom_distance_cutoff=4.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)

    if binder_chain not in structure[0]:
        print(f"警告：在 {trajectory_pdb} 中未找到Binder链 '{binder_chain}'。")
        return {}
        
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], 'A')
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    if 'B' not in structure[0]:
        print(f"警告：在 {trajectory_pdb} 中未找到靶标链 'B'。")
        return {}
        
    target_atoms = Selection.unfold_entities(structure[0]['B'], 'A')
    target_coords = np.array([atom.coord for atom in target_atoms])

    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)
    interacting_residues = {}
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        if binder_resname in three_to_one_map:
            aa_single_letter = three_to_one_map[binder_resname]
            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues

####################################################################
# 核心函数 (来自 pyrosetta_utils.py)
# [函数 'score_interface' 在此省略，与上一版相同]
####################################################################

def score_interface(pdb_file, binder_chain="A"):
    pose = pr.pose_from_pdb(pdb_file)
    iam = InterfaceAnalyzerMover()
    iam.set_interface(f"{binder_chain}_B") 
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    interface_AA = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    interface_residues_set = hotspot_residues(pdb_file, binder_chain)
    interface_residues_pdb_ids = []
    
    for pdb_res_num, aa_type in interface_residues_set.items():
        if aa_type in interface_AA:
            interface_AA[aa_type] += 1
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")

    interface_nres = len(interface_residues_pdb_ids)
    interface_residues_pdb_ids_str = ','.join(interface_residues_pdb_ids)

    hydrophobic_aa = set('ACFILMPVWY')
    hydrophobic_count = sum(interface_AA[aa] for aa in hydrophobic_aa)
    if interface_nres != 0:
        interface_hydrophobicity = (hydrophobic_count / interface_nres) * 100
    else:
        interface_hydrophobicity = 0.0

    interfacescore = iam.get_all_data()
    interface_sc = interfacescore.sc_value
    interface_interface_hbonds = interfacescore.interface_hbonds
    interface_dG = iam.get_interface_dG()
    interface_dSASA = iam.get_interface_delta_sasa()
    interface_packstat = iam.get_interface_packstat()
    interface_dG_SASA_ratio = interfacescore.dG_dSASA_ratio * 100
    
    buns_filter = XmlObjects.static_get_filter('<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />')
    interface_delta_unsat_hbonds = buns_filter.report_sm(pose)
    
    if interface_nres != 0:
        interface_hbond_percentage = (interface_interface_hbonds / interface_nres) * 100
        interface_bunsch_percentage = (interface_delta_unsat_hbonds / interface_nres) * 100
    else:
        interface_hbond_percentage = 0.0
        interface_bunsch_percentage = 0.0

    chain_design = ChainSelector(binder_chain)
    tem = TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = tem.calculate(pose)

    bsasa = SasaMetric()
    bsasa.set_residue_selector(chain_design)
    binder_sasa = bsasa.calculate(pose)

    if binder_sasa > 0:
        interface_binder_fraction = (interface_dSASA / binder_sasa) * 100
    else:
        interface_binder_fraction = 0.0

    binder_chain_index = -1
    for i in range(1, pose.num_chains() + 1):
        if pose.pdb_info().chain(pose.conformation().chain_begin(i)) == binder_chain:
            binder_chain_index = i
            break
            
    if binder_chain_index == -1:
        surface_hydrophobicity = 0.0
    else:
        binder_pose = pose.split_by_chain(binder_chain_index)
        layer_sel = LayerSelector()
        layer_sel.set_layers(pick_core = False, pick_boundary = False, pick_surface = True)
        surface_res = layer_sel.apply(binder_pose)

        exp_apol_count = 0
        total_count = 0 
        for i in range(1, len(surface_res) + 1):
            if surface_res[i] == True:
                res = binder_pose.residue(i)
                if res.is_apolar() or res.name() == 'PHE' or res.name() == 'TRP' or res.name() == 'TYR':
                    exp_apol_count += 1
                total_count += 1
        
        if total_count > 0:
            surface_hydrophobicity = exp_apol_count / total_count
        else:
            surface_hydrophobicity = 0.0

    interface_scores = {
        'binder_score': binder_score,
        'surface_hydrophobicity': surface_hydrophobicity,
        'interface_sc': interface_sc,
        'interface_packstat': interface_packstat,
        'interface_dG': interface_dG,
        'interface_dSASA': interface_dSASA,
        'interface_dG_SASA_ratio': interface_dG_SASA_ratio,
        'interface_fraction': interface_binder_fraction,
        'interface_hydrophobicity': interface_hydrophobicity,
        'interface_nres': interface_nres,
        'interface_interface_hbonds': interface_interface_hbonds,
        'interface_hbond_percentage': interface_hbond_percentage,
        'interface_delta_unsat_hbonds': interface_delta_unsat_hbonds,
        'interface_delta_unsat_hbonds_percentage': interface_bunsch_percentage
    }

    interface_scores = {k: round(v, 2) if isinstance(v, (float, np.floating)) else v for k, v in interface_scores.items()}

    return interface_scores, interface_AA, interface_residues_pdb_ids_str

####################################################################
# 脚本执行入口
####################################################################

if __name__ == "__main__":
    # --- 1. 设置参数解析 ---
    parser = argparse.ArgumentParser(
        description="批量计算PDB复合物的界面分数并保存到CSV (带即时耗时)。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # [参数解析代码与上一版相同，在此省略]
    parser.add_argument(
        "-i", "--input_dir", 
        default="./pdbs_scores",
        help="包含PDB文件的输入目录 (默认: ./pdbs_scores)"
    )
    parser.add_argument(
        "-o", "--output_csv", 
        default="interface_scores_summary.csv",
        help="输出CSV文件的路径 (默认: interface_scores_summary.csv)"
    )
    parser.add_argument(
        "-chain", "--binder_chain", 
        default="A", 
        help="作为 'binder' 的链ID (默认: A)。\n脚本假设另一条链是 'B'。"
    )
    parser.add_argument(
        "-dalphaball", "--dalphaball_path",
        required=True,
        help="DAlphaBall.gcc 或 surf_vol 可执行文件的**完整路径**。\n(这是计算 BuriedUnsatHbonds 所必需的)"
    )
    args = parser.parse_args()


    # --- 2. 检查 DAlphaBall 路径 ---
    if not os.path.exists(args.dalphaball_path):
        print(f"错误: DAlphaBall 可执行文件未找到: {args.dalphaball_path}")
        print("请使用 --dalphaball_path 提供正确的路径。")
        exit(1)

    # --- 3. 初始化 PyRosetta ---
    print("正在初始化 PyRosetta...")
    init_flags = (
        "-ignore_unrecognized_res "
        "-load_PDB_components false "
        "-holes:dalphaball "
        f"-dalphaball {args.dalphaball_path}"
    )
    pr.init(init_flags) 

    # --- 4. 查找 PDB 文件 ---
    search_pattern = os.path.join(args.input_dir, "*_model.pdb")
    pdb_files = glob.glob(search_pattern)

    if not pdb_files:
        print(f"错误: 在 '{args.input_dir}' 中未找到匹配 '*_model.pdb' 的文件。")
        exit(1)

    total_files = len(pdb_files)
    print(f"找到了 {total_files} 个PDB文件。开始处理...")

    # --- 5. 循环处理文件 ---
    all_results = []
    total_start_time = time.time() # (可选) 记录总开始时间

    for i, pdb_path in enumerate(pdb_files):
        file_basename = os.path.basename(pdb_path)
        print(f"--- 处理文件 {i+1}/{total_files}: {file_basename} ---")
        
        # <-- 2. 记录单个文件开始时间
        file_start_time = time.time() 
        
        try:
            # 提取 'name'
            name = file_basename.rsplit('_model.pdb', 1)[0]
            
            # 计算分数
            (
                scores, 
                aa_counts, 
                interface_residues_str
            ) = score_interface(pdb_path, args.binder_chain)
            
            # <-- 3. 记录结束时间并计算耗时
            file_end_time = time.time()
            duration = file_end_time - file_start_time
            print(f"    > 完成。耗时: {duration:.2f} 秒。") # <-- 这是您需要的新增输出
            
            # --- 准备要写入CSV的数据 ---
            row_data = {}
            row_data['name'] = name
            row_data['processing_time_s'] = round(duration, 2) # (可选) 将耗时也存入CSV
            row_data.update(scores)
            row_data['interface_residues_str'] = interface_residues_str
            flat_aa_counts = {f"AA_{aa}": count for aa, count in aa_counts.items()}
            row_data.update(flat_aa_counts)
            
            all_results.append(row_data)

        except Exception as e:
            # <-- 4. 即使失败也记录耗时
            file_end_time = time.time()
            duration = file_end_time - file_start_time
            print(f"\n[警告] 处理文件 {pdb_path} 失败: {e} (耗时 {duration:.2f} 秒)")
            print("该文件将被跳过。")

    # --- 6. 保存到 CSV ---
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\n处理完成。总共 {len(all_results)} / {total_files} 个文件成功。")
    print(f"总耗时: {total_duration:.2f} 秒。")
    
    if not all_results:
        print("未成功处理任何文件。")
        exit(1)
        
    df = pd.DataFrame(all_results)
    
    # 重新排序列
    key_cols = ['name', 'processing_time_s', 'interface_dG', 'interface_dSASA', 'interface_sc', 'interface_packstat', 'interface_nres']
    existing_key_cols = [col for col in key_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in existing_key_cols]
    
    df = df[existing_key_cols + sorted(other_cols)]

    df.to_csv(args.output_csv, index=False)
    
    print(f"结果已保存到: {args.output_csv}")