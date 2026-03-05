
import os
import sys
import time
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_planner import DiffusionPlanner
from maze_utils import (
    generate_kruskal_maze,
    maze_to_model_input,
    compute_path_metrics,
)

DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
DIFFUSION_ITERS = 100  # Max = scheduler num_train_timesteps
PATH_STEPS = 64


def plot_path(ax, maze_img, path, start, goal, title=''):
    disp = 1 - (maze_img / 255.0) if maze_img.max() > 1 else 1 - maze_img
    ax.imshow(disp, cmap='gray', vmin=0, vmax=1)
    px = (np.array(path)[:, 0] + 5) / 0.1
    py = (5 - np.array(path)[:, 1]) / 0.1
    ax.plot(px, py, 'r-', linewidth=2, label='Path')
    ax.scatter(px[0], py[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(px[-1], py[-1], c='blue', s=100, marker='x', linewidths=3, label='Goal', zorder=5)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='./outputs', help='Directory for metrics and figures')
    parser.add_argument('--ckpt', default=None, help='Path to checkpoint (default: checkpoints/dipper_final.pth)')
    parser.add_argument('--data-dir', default=None,
                        help='Data folder (default: ./data/data_20230908-113959_10k_maps_100_traj)')
    parser.add_argument('--maze-paths', nargs='*', default=None,
                        help='Optional: override mazes with explicit image paths (e.g., data/infer/kruskal_1.png data/infer/kruskal_2.png)')
    parser.add_argument('--diffusion-iters', type=int, default=DIFFUSION_ITERS)
    parser.add_argument('--path-steps', type=int, default=PATH_STEPS)

    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"Device: {DEVICE}\n")

    # 2 Kruskal mazes only - diffusion path planning showcase
    # If --maze-paths is provided, those images are used for inference.
    if args.maze_paths:
        import cv2
        MAZES = []
        for mp in args.maze_paths:
            img = cv2.imread(mp, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read maze image: {mp}")
            name = os.path.splitext(os.path.basename(mp))[0]
            # Default start/goal pairs (world coords) unless user edits below
            default_pairs = [((-4, -4), (4, 4))]
            MAZES.append((name, img, default_pairs))
        print(f"Loaded mazes from explicit paths: {args.maze_paths}")
    else:
        data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), 'data', 'data_20230908-113959_10k_maps_100_traj')
        if os.path.exists(data_dir):
            import cv2
            m1 = cv2.imread(os.path.join(data_dir, 'maze_occu_0.png'))
            m2 = cv2.imread(os.path.join(data_dir, 'maze_occu_1.png'))
            if m1 is not None and m2 is not None:
                MAZES = [
                    ('Kruskal_1', m1, [((-4, -4), (4, 4)), ((-3, 3), (3, -3))]),
                    ('Kruskal_2', m2, [((-4, 4), (4, -4))]),
                ]
                print(f'Loaded mazes from {data_dir}')
            else:
                MAZES = [
                    ('Kruskal_1', generate_kruskal_maze(seed=42), [((-4, -4), (4, 4)), ((-3, 3), (3, -3))]),
                    ('Kruskal_2', generate_kruskal_maze(seed=123), [((-4, 4), (4, -4))]),
                ]
        else:
            MAZES = [
                ('Kruskal_1', generate_kruskal_maze(seed=42), [((-4, -4), (4, 4)), ((-3, 3), (3, -3))]),
                ('Kruskal_2', generate_kruskal_maze(seed=123), [((-4, 4), (4, -4))]),
            ]

    # Load planner
    planner = DiffusionPlanner(device=DEVICE)
    ckpt = args.ckpt or os.path.join(os.path.dirname(__file__), 'checkpoints', 'dipper_final.pth')
    if os.path.exists(ckpt):
        planner.load_weights(ckpt)
        print("Loaded pretrained weights.\n")
    else:
        print("No checkpoint found – using random weights (paths may be invalid).\n")

    # Run planning
    results = []
    for maze_name, maze_img, start_goals in MAZES:
        for (start, goal) in start_goals:
            map_input = maze_to_model_input(maze_img)
            t0 = time.perf_counter()
            path = planner.plan(map_input, np.array(start), np.array(goal),
                               diffusion_iters=args.diffusion_iters, path_steps=args.path_steps)
            elapsed = time.perf_counter() - t0

            metrics = compute_path_metrics(path, maze_img)
            metrics['inference_time_s'] = elapsed
            results.append({
                'maze': maze_name,
                'maze_img': maze_img,
                'start': list(start),
                'goal': list(goal),
                'path': path,
                'metrics': metrics,
            })
            print(f"{maze_name} | {start} -> {goal} | {elapsed:.3f}s | "
                  f"{metrics['path_length_m']:.2f}m | Feasible: {metrics['feasible']}")

    # Save metrics JSON
    metrics_summary = [
        {
            'maze': r['maze'],
            'start': r['start'],
            'goal': r['goal'],
            'inference_time_s': r['metrics']['inference_time_s'],
            'path_length_m': r['metrics']['path_length_m'],
            'n_waypoints': r['metrics']['n_waypoints'],
            'feasible': r['metrics']['feasible'],
            'collision_rate': r['metrics']['collision_rate'],
        }
        for r in results
    ]
    metrics_path = os.path.join(out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Save summary stats
    avg_time = np.mean([r['metrics']['inference_time_s'] for r in results])
    pct_feasible = 100 * np.mean([r['metrics']['feasible'] for r in results])
    summary = {
        'n_plans': len(results),
        'avg_inference_time_s': float(avg_time),
        'feasibility_rate_pct': float(pct_feasible),
        'diffusion_iters': int(args.diffusion_iters),
        'path_steps': int(args.path_steps),
    }
    summary_path = os.path.join(out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    # Save metrics CSV
    import csv
    csv_path = os.path.join(out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['maze', 'start', 'goal', 'inference_time_s', 'path_length_m', 'n_waypoints', 'feasible'])
        w.writeheader()
        for row in metrics_summary:
            w.writerow({k: row[k] for k in w.fieldnames})
    print(f"Saved CSV to {csv_path}")

    # Save Markdown report
    md_path = os.path.join(out_dir, 'REPORT.md')
    with open(md_path, 'w') as f:
        f.write("# DiPPeR Diffusion Path Planning - Results\n\n")
        f.write("## Summary\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Plans executed | {summary['n_plans']} |\n")
        f.write(f"| Avg inference time | {summary['avg_inference_time_s']:.3f} s |\n")
        f.write(f"| Feasibility rate | {summary['feasibility_rate_pct']:.1f}% |\n")
        f.write(f"| Diffusion iterations | {summary['diffusion_iters']} |\n")
        f.write(f"| Path waypoints | {summary['path_steps']} |\n\n")
        f.write("## Per-Plan Metrics\n\n")
        f.write("| Maze | Start | Goal | Time (s) | Path (m) | Feasible |\n")
        f.write("|------|-------|------|----------|----------|----------|\n")
        for row in metrics_summary:
            f.write(f"| {row['maze']} | {row['start']} | {row['goal']} | "
                    f"{row['inference_time_s']:.3f} | {row['path_length_m']:.2f} | {row['feasible']} |\n")
        f.write("\n## Outputs\n\n")
        f.write("- `metrics.json` - Full metrics\n- `metrics.csv` - Tabular metrics\n- `summary.json` - Aggregate stats\n")
        f.write("- `planned_paths.png` - Path visualizations\n- `mazes.png` - Input mazes\n")
    print(f"Saved report to {md_path}")

    print(f"\n--- Summary ---")
    print(f"Average inference time: {avg_time:.3f} s")
    print(f"Feasibility rate: {pct_feasible:.1f}%")

    # Save figures
    n_results = len(results)
    n_cols = 2
    n_rows = (n_results + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    if n_rows == 1:
        axes = np.array([axes]) if n_cols == 1 else axes.reshape(1, -1)
    else:
        axes = axes.reshape(-1, n_cols) if axes.ndim == 1 else axes

    for idx, r in enumerate(results):
        ax = axes.flat[idx]
        m = r['metrics']
        title = f"{r['maze']} | {r['start']} → {r['goal']} | Feasible: {m['feasible']} | {m['inference_time_s']:.2f}s"
        plot_path(ax, r['maze_img'], r['path'], r['start'], r['goal'], title=title)

    for j in range(idx + 1, len(axes.flat)):
        axes.flat[j].axis('off')

    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'planned_paths.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {fig_path}")

    # Maze overview
    n_mazes = len(MAZES)
    fig2, axes2 = plt.subplots(1, n_mazes, figsize=(5 * n_mazes, 5))
    if n_mazes == 1:
        axes2 = [axes2]
    for ax, (name, maze, _) in zip(axes2, MAZES):
        ax.imshow(maze, cmap='gray', vmin=0, vmax=255)
        ax.set_title(name)
        ax.axis('off')
    plt.tight_layout()
    fig2_path = os.path.join(out_dir, 'mazes.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved mazes to {fig2_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
