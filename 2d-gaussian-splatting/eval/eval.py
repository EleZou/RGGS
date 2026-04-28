import os
import sys
from mesh_eval import eval_mesh, to_paper_metrics
from image_eval import eval_images
from argparse import ArgumentParser
import json
import torch
from glob import glob


PAPER_METRIC_ORDER = ('CD', 'N_GS', 'Mem', 'PSNR', 'SSIM', 'LPIPS')


def find_existing_path(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


def get_point_cloud_root(model_path):
    return find_existing_path([
        os.path.join(model_path, 'free_gaussians', 'point_cloud'),
        os.path.join(model_path, 'free_gaussians_reinforce', 'point_cloud'),
    ])


def get_mesh_root(model_path):
    return find_existing_path([
        os.path.join(model_path, 'tetra_meshes'),
        os.path.join(model_path, 'tetra_meshes_reinforce'),
    ])


def get_render_root(model_path, iteration):
    free_gaussians_root = find_existing_path([
        os.path.join(model_path, 'free_gaussians'),
        os.path.join(model_path, 'free_gaussians_reinforce'),
    ])
    return os.path.join(free_gaussians_root, 'train', f'ours_{iteration}', 'renders')


def count_gaussians(point_cloud_path):
    with open(point_cloud_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('element vertex'):
                return int(line.strip().split()[-1])
            if line.strip() == 'end_header':
                break
    raise ValueError(f"Failed to read Gaussian count from PLY header: {point_cloud_path}")


if __name__ == '__main__':

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Eval Replica Scene")
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--iteration", type=str, default='-1')
    parser.add_argument("--sparse_view_num", type=str, required=True)
    parser.add_argument("--eval_obj_mesh", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    iter = int(args.iteration)
    point_cloud_root_path = get_point_cloud_root(args.model_path)
    if iter == -1:
        iter_file_list = os.listdir(point_cloud_root_path)
        iter_list = [int(file.split('_')[-1]) for file in iter_file_list]
        max_iter = max(iter_list)
        iter = max_iter

    print(f"Evaluating iteration {iter}")
    sparse_view_num = args.sparse_view_num

    metrics = {}
    point_cloud_path = os.path.join(point_cloud_root_path, f'iteration_{iter}', 'point_cloud.ply')
    metrics['N_GS'] = round(count_gaussians(point_cloud_path) / 1_000_000, 5)

    # eval mesh
    pred_scene_mesh_root_path = get_mesh_root(args.model_path)
    file_list = os.listdir(pred_scene_mesh_root_path)
    iter_file_list = [file for file in file_list if f'_iter_{iter}.ply' in file]
    assert len(iter_file_list) == 1, f"Found {len(iter_file_list)} meshes for iteration {iter}"

    pred_mesh_path = os.path.join(pred_scene_mesh_root_path, iter_file_list[0])
    metrics['Mem'] = round(os.path.getsize(pred_mesh_path) / (1024 * 1024), 5)
    gt_mesh_path = os.path.join(args.source_path, 'gt_mesh', 'scene_mesh.ply')
    if os.path.exists(gt_mesh_path):
        mesh_metrics = eval_mesh(pred_mesh_path, gt_mesh_path)
        metrics.update(to_paper_metrics(mesh_metrics))
    else:
        print(f"No gt mesh found at {gt_mesh_path}")

    # eval obj mesh
    if args.eval_obj_mesh:
        pred_obj_mesh_list = glob(os.path.join(args.model_path, 'mesh', f'obj_*_{iter}.ply'))
        pred_obj_mesh_list.sort()
        for pred_obj_mesh_path in pred_obj_mesh_list:
            obj_mesh_name = os.path.basename(pred_obj_mesh_path)
            obj_id = int(obj_mesh_name.split('_')[-2])
            gt_obj_mesh_path = os.path.join(args.source_path, 'gt_mesh', f'obj_{obj_id}.ply')
            obj_mesh_metrics = eval_mesh(pred_obj_mesh_path, gt_obj_mesh_path)
            obj_metrics = to_paper_metrics(obj_mesh_metrics)
            save_obj_mesh_path = os.path.join(args.model_path, f'obj_{obj_id}_{iter}_paper_metrics.json')
            with open(save_obj_mesh_path, 'w') as f:
                json.dump(obj_metrics, f, indent=2)
            print(f"obj {obj_id} eval done")

    # eval images
    split_file = os.path.join(args.source_path, f'split-{sparse_view_num}views.json')
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            split_dict = json.load(f)
        test_views_list = split_dict['test']
    else:
        split_file = os.path.join(args.source_path, f'train_test_split_{sparse_view_num}.json')
        with open(split_file, 'r') as f:
            split_dict = json.load(f)
        test_views_list = split_dict['test_ids']
    test_views_list.sort()

    pred_img_root_path = get_render_root(args.model_path, iter)
    ssims, psnrs, lpipss = eval_images(args.source_path, pred_img_root_path, test_views_list)
    metrics['PSNR'] = round(torch.tensor(psnrs).mean().item(), 5)
    metrics['SSIM'] = round(torch.tensor(ssims).mean().item(), 5)
    metrics['LPIPS'] = round(torch.tensor(lpipss).mean().item(), 5)
    metrics = {key: metrics[key] for key in PAPER_METRIC_ORDER if key in metrics}

    print(f"Metrics for {args.model_path}:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    result_file = os.path.join(args.model_path, f'result_iter_{iter}.txt')
    with open(result_file, 'w') as f:
        f.write(f"Metrics for {args.model_path}:\n")
        for k, v in metrics.items():
            out = f"{k}: {v}\n"
            f.write(out)

    result_file = os.path.join(args.model_path, f'result_iter_{iter}.json')
    with open(result_file, 'w') as f:
        json.dump(metrics, f)
    print(f"Save mesh metrics to {result_file}")
