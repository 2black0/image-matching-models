"""
This script performs image matching using a specified matcher model. It processes pairs of input images,
detects keypoints, matches them, and performs RANSAC to find inliers. The results, including visualizations
and metadata, are saved to the specified output directory.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch

import os
import sys
# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import traceback
from pathlib import Path
import json
import numpy as np
import textwrap


from matching.utils import get_image_pairs_paths, get_default_device
from matching import get_matcher, available_models
from matching.viz import plot_matches


COL_WIDTH = 22


def parse_args():
    # Format available matchers in columns, shown at the end of the help message (python imm_match.py -h)
    matchers = sorted(available_models)
    matcher_list_str = ", ".join(matchers)
    wrapped_matchers = textwrap.fill(matcher_list_str, width=80, initial_indent="  ", subsequent_indent="  ")

    parser = argparse.ArgumentParser(
        prog="imm-match",
        description="Match keypoints between image pairs. Outputs match visualizations and result dicts.",
        epilog=f"Available matchers ({len(matchers)}):\n" + wrapped_matchers,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="sift-lightglue",
        choices=available_models + ["all"],
        metavar="MODEL",
        help="matcher to use (default: %(default)s). See list below",
    )

    # Hyperparameters shared by all methods:
    parser.add_argument("--im_size", type=int, default=512, help="resize img to im_size x im_size")
    parser.add_argument("--n_kpts", type=int, default=2048, help="max num keypoints")
    parser.add_argument("--device", type=str, default=get_default_device(), choices=["cpu", "cuda"])
    parser.add_argument("--outlier", type=str, default="cv-magsac", help="RANSAC method: cv-ransac, cv-magsac, cv-gc-ransac, cv-usac-fast, pl-lo-ransac, pl-prosac, degensac")
    parser.add_argument("--no_viz", action="store_true", help="avoid saving visualizations")

    parser.add_argument("--max_pairs", type=int, default=None, help="maximum number of pairs to process")

    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",  # Accept one or more arguments
        default=[Path(os.path.join(os.path.dirname(__file__), "../assets/example_pairs"))],
        help="path to either (1) two image paths or (2) dir with two images or (3) dir with dirs with image pairs or "
        "(4) txt file with two image paths per line",
    )
    parser.add_argument("--out_dir", type=Path, default=None, help="path where outputs are saved")

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(os.path.join(os.path.dirname(__file__), "../outputs")) / args.matcher

    return args


def main():
    args = parse_args()
    image_size = [args.im_size, args.im_size]
    
    # Handle "all" matchers
    if args.matcher == 'all':
        # Skip heavy/OOM matchers as requested
        skip_matchers = [
            # Heavy/OOM matchers
            'roma', 'romav2', 'tiny-roma',
            'duster', 'master', 
            'gim-dkm',
            'ufm',
            # Problematic matchers with persistent namespace collisions
            'matchanything-roma',
            'minima-roma',
           'omniglue', 'omniglue-onnx',
            # Toolbox matchers (tools module collision from MatchAnything)
            'patch2pix', 'superglue', 'r2d2', 'd2net', 'doghardnet-nn',
            # SphereGlue-based matchers
            'sift-sphereglue', 'superpoint-sphereglue',
            # MINIMA-based matchers
            'minima', 'minima-roma-tiny', 'minima-superpoint-lightglue',
            'minima-loftr', 'minima-xoftr'
        ]
        matchers_to_run = [m for m in available_models if m not in skip_matchers]
    else:
        matchers_to_run = [args.matcher]

    for matcher_name in matchers_to_run:
        print(f"\nProcessing with matcher: {matcher_name}")
        
        # Create timestamped output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Override out_dir if not specified, otherwise append matcher name
        if args.out_dir is None:
             base_out_dir = Path(os.path.join(os.path.dirname(__file__), "../outputs"))
             current_out_dir = base_out_dir / f"{matcher_name}_{timestamp}"
        else:
             # If user specified out_dir, we follow the structure but make it unique per matcher run if multiple
             if len(matchers_to_run) > 1:
                 current_out_dir = args.out_dir / f"{matcher_name}_{timestamp}"
             else:
                 current_out_dir = args.out_dir

        current_out_dir.mkdir(exist_ok=True, parents=True)

        try:
            # Choose a matcher
            matcher = get_matcher(matcher_name, device=args.device, max_num_keypoints=args.n_kpts, outlier=args.outlier)
            print(f"Using matcher: {matcher_name} on device: {args.device} with outlier method: {args.outlier}")
            print("=" * 80)

            pairs_of_paths = get_image_pairs_paths(args.input)
            for i, (img0_path, img1_path) in enumerate(pairs_of_paths):
                if args.max_pairs is not None and i >= args.max_pairs:
                    print(f"Reached max pairs limit ({args.max_pairs}). Stopping.")
                    break
                start = time.time()
                image0 = matcher.load_image(img0_path, resize=image_size)
                image1 = matcher.load_image(img1_path, resize=image_size)
                result = matcher(image0, image1)

                out_str = f"{'Paths':<{COL_WIDTH}}: {img0_path}, {img1_path}\n"
                out_str += f"{'Inliers (post-RANSAC)':<{COL_WIDTH}}: {result['num_inliers']}\n"

                if not args.no_viz:
                    viz_path = current_out_dir / f"output_{i}_matches.jpg"
                    plot_matches(image0, image1, result, save_path=viz_path)
                    import matplotlib.pyplot as plt
                    plt.close('all')
                    out_str += f"{'Viz saved in':<{COL_WIDTH}}: {viz_path}\n"

                result["img0_path"] = str(img0_path)
                result["img1_path"] = str(img1_path)
                result["matcher"] = matcher_name
                result["n_kpts"] = args.n_kpts
                result["im_size"] = args.im_size

                dict_path = current_out_dir / f"output_{i}_result.torch"
                torch.save(result, dict_path)
                out_str += f"{'Output saved in':<{COL_WIDTH}}: {dict_path}\n"
                out_str += f"{'Time taken (s)':<{COL_WIDTH}}: {time.time() - start:.3f}\n"

                # Save as JSON
                json_path = current_out_dir / f"output_{i}_matches.json"
                
                # Prepare dict for JSON serialization
                inference_time = time.time() - start
                json_result = {}
                
                # List of keys requested by user
                keys_to_save = [
                    "num_inliers", "H", "all_kpts0", "all_kpts1", 
                    "all_desc0", "all_desc1", "matched_kpts0", 
                    "matched_kpts1", "inlier_kpts0", "inlier_kpts1"
                ]
                
                for k in keys_to_save:
                    val = result.get(k, None)
                    if isinstance(val, np.ndarray):
                        json_result[k] = val.tolist()
                    elif isinstance(val, torch.Tensor):
                        json_result[k] = val.cpu().numpy().tolist()
                    else:
                        json_result[k] = val
                        
                json_result["inference_time"] = inference_time
                
                with open(json_path, 'w') as f:
                    json.dump(json_result, f, indent=4)
                    
                out_str += f"{'JSON saved in':<{COL_WIDTH}}: {json_path}\n"

                print(out_str)
        except Exception as e:
            print(f"Error running matcher {matcher_name}: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
