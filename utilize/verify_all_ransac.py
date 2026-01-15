
import numpy as np
import cv2
import time
import argparse
import sys
import math


try:
    import poselib
    HAS_POSELIB = True
except ImportError:
    print("PoseLib not found")
    HAS_POSELIB = False


try:
    import pydegensac
    HAS_DEGENSAC = True
except ImportError:
    print("pydegensac not found")
    HAS_DEGENSAC = False

try:
    import pygcransac
    HAS_GCRANSAC = True
except ImportError:
    print("pygcransac not found")
    HAS_GCRANSAC = False

def generate_data(n_inliers=100, n_outliers=50, noise=0.5, img_size=(1000, 1000)):
    # Ground truth homography (mostly identity with slight warp)
    H_gt = np.eye(3)
    H_gt[0, 2] = 50
    H_gt[1, 2] = -30
    H_gt[0, 0] = 1.05
    
    # Inliers
    pts0_in = np.random.rand(n_inliers, 2) * np.array(img_size)
    pts0_in_h = np.concatenate([pts0_in, np.ones((n_inliers, 1))], axis=1)
    pts1_in_h = (H_gt @ pts0_in_h.T).T
    pts1_in = pts1_in_h[:, :2] / pts1_in_h[:, 2:]
    
    # Add noise
    pts1_in += np.random.normal(0, noise, pts1_in.shape)
    
    # Outliers
    pts0_out = np.random.rand(n_outliers, 2) * np.array(img_size)
    pts1_out = np.random.rand(n_outliers, 2) * np.array(img_size)
    
    pts0 = np.concatenate([pts0_in, pts0_out])
    pts1 = np.concatenate([pts1_in, pts1_out])
    
    # Shuffle
    idx = np.arange(len(pts0))
    np.random.shuffle(idx)
    return pts0[idx], pts1[idx], img_size

def run_opencv(method_name, flag, pts0, pts1, threshold=3.0, loops=1):
    times = []
    H_final = None
    inliers_final = 0
    
    for _ in range(loops):
        start = time.time()
        H, mask = cv2.findHomography(pts0, pts1, flag, threshold)
        dur = (time.time() - start) * 1000
        times.append(dur)
        H_final = H
        inliers_final = np.sum(mask.ravel()) if mask is not None else 0

    return H_final, inliers_final, times



def run_degensac(pts0, pts1, threshold=3.0, loops=1):
    # pydegensac.findHomography(src_pts, dst_pts, th, conf, n_iter)
    # It requires float32 input
    pts0 = pts0.astype(np.float32)
    pts1 = pts1.astype(np.float32)
    
    times = []
    H_final = None
    inliers_final = 0
    
    for _ in range(loops):
        start = time.time()
        H, mask = pydegensac.findHomography(pts0, pts1, threshold, 0.99, 2000)
        dur = (time.time() - start) * 1000
        times.append(dur)
        H_final = H
        inliers_final = int(mask.sum()) if mask is not None else 0
        
    return H_final, inliers_final, times

def run_poselib(pts0, pts1, img_size, threshold=3.0, prosac=False, loops=1):
    # Required options
    ransac_opt = {'max_iterations': 1000, 'success_prob': 0.9999, 'max_reproj_error': threshold}
    
    if prosac:
        ransac_opt['progressive_sampling'] = True

    times = []
    H_final = None
    inliers_final = 0

    for _ in range(loops):
        start = time.time()
        result = poselib.estimate_homography(pts0, pts1, ransac_opt)
        dur = (time.time() - start) * 1000
        times.append(dur)
        
        if result is None:
            H_final = None
            inliers_final = 0
            continue
            
        if isinstance(result, tuple):
             H_final = result[0]
             inliers_final = sum(result[1]['inliers']) if 'inliers' in result[1] else 0
        elif hasattr(result, 'H'):
             H_final = result.H
             inliers_final = np.sum(result.inliers)
        else:
             print(f"Unknown PoseLib return type: {type(result)}")
             H_final = None
             inliers_final = 0

    return H_final, inliers_final, times


def print_result(name, n, times, h_valid):
    if times is None:
        print(f"{name:<20} | {'ERR':<10} | {'N/A':<9} | {'N/A':<9} | {'N/A':<9} | N/A")
        return

    avg = np.mean(times)
    min_t = np.min(times)
    max_t = np.max(times)
    std = np.std(times)
    
    print(f"{name:<20} | {n:<10} | {avg:<9.2f} | {min_t:<9.2f} | {max_t:<9.2f} | {h_valid}")


def run_gcransac(pts0, pts1, img_size, threshold=3.0, affine=False, loops=1):
    # Initializes probabilities for PROSAC sampler (optional, but good for benchmarks)
    # Using uniform probabilities for fair comparison with basic OpenCV methods if not specified
    
    # Needs stacked input [x1, y1, x2, y2, ...]
    corresp = np.hstack([pts0, pts1])
    
    # Solver flag: 0=point (default), 2=affine
    solver_flag = 2 if affine else 0
    
    times = []
    H_final = None
    inliers_final = 0

    for _ in range(loops):
        start = time.time()
        # pygcransac expects float64 contiguous arrays
        kp1 = corresp[:, :2].astype(np.float64)
        kp2 = corresp[:, 2:].astype(np.float64)
        # It actually takes a single contiguous array in some versions, but let's check the call signature used previously
        # The binding usually expects one big array (N, 4)
        corresp_contiguous = np.ascontiguousarray(corresp, dtype=np.float64)
        
        # DEBUG: Print types to diagnose mismatch
        # print(f"DEBUG: Shape={corresp_contiguous.shape}, Dtype={corresp_contiguous.dtype}, C_Contiguous={corresp_contiguous.flags['C_CONTIGUOUS']}")

        try:
            # Note: Signature requires 'probabilities' argument after w2
            # findHomography(correspondences, h1, w1, h2, w2, probabilities, ...)
            H, mask = pygcransac.findHomography(
                corresp_contiguous, 
                int(img_size[0]), int(img_size[1]), 
                int(img_size[0]), int(img_size[1]),
                [], # probabilities (empty list = uniform)
                threshold = threshold,
                conf = 0.9999,
                max_iters = 1000,
                solver = solver_flag,
                sampler = 1 # PROSAC
            )
        except TypeError as e:
             # Fallback debugging if signature mismatch
             print(f"DEBUG GCRANSAC: {e}")
             return None, 0, []
             
        dur = (time.time() - start) * 1000
        times.append(dur)
        
        H_final = H
        inliers_final = int(mask.sum()) if mask is not None else 0

    return H_final, inliers_final, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=1, help="Number of loops for averaging time")
    parser.add_argument("--inliers", type=int, default=500, help="Number of inliers")
    parser.add_argument("--outliers", type=int, default=200, help="Number of outliers")
    parser.add_argument("--noise", type=float, default=0.5, help="Standard deviation of noise (pixels)")
    parser.add_argument("--threshold", type=float, default=3.0, help="RANSAC reprojection threshold")
    
    args = parser.parse_args()
    
    print(f"OpenCV Version: {cv2.__version__}")
    
    
    pts0, pts1, size = generate_data(n_inliers=args.inliers, n_outliers=args.outliers, noise=args.noise)
    print(f"Data: {len(pts0)} matches ({args.inliers} inliers gt), Noise: {args.noise}px")
    print(f"Loops: {args.loop}, Threshold: {args.threshold}px")
    print("-" * 85)
    print(f"{'Method':<20} | {'Inliers':<10} | {'Avg(ms)':<9} | {'Min(ms)':<9} | {'Max(ms)':<9} | {'H Valid':<10}")
    print("-" * 85)
    
    # 1. OpenCV RANSAC
    H, n, t = run_opencv("CV_RANSAC", cv2.RANSAC, pts0, pts1, threshold=args.threshold, loops=args.loop)
    print_result("cv-ransac", n, t, H is not None)

    # 2. OpenCV MAGSAC++
    H, n, t = run_opencv("CV_MAGSAC++", cv2.USAC_MAGSAC, pts0, pts1, threshold=args.threshold, loops=args.loop)
    print_result("cv-magsac", n, t, H is not None)
    
    # 3. OpenCV GC-RANSAC (USAC_ACCURATE)
    try:
        H, n, t = run_opencv("CV_GC-RANSAC", cv2.USAC_ACCURATE, pts0, pts1, threshold=args.threshold, loops=args.loop)
        print_result("cv-gc-ransac", n, t, H is not None)
    except AttributeError:
        print_result("cv-gc-ransac", "N/A", None, "N/A")

    # 4. OpenCV USAC_FAST
    try:
        H, n, t = run_opencv("CV_USAC_FAST", cv2.USAC_FAST, pts0, pts1, threshold=args.threshold, loops=args.loop)
        print_result("cv-usac-fast", n, t, H is not None)
    except AttributeError:
        print_result("cv-usac-fast", "N/A", None, "N/A")

    # 5. OpenCV RHO
    try:
        H, n, t = run_opencv("CV_RHO", cv2.RHO, pts0, pts1, threshold=args.threshold, loops=args.loop)
        print_result("cv-rho", n, t, H is not None)
    except AttributeError:
        print_result("cv-rho", "N/A", None, "N/A")
    
    # 6. PoseLib (Lo-RANSAC)
    if HAS_POSELIB:
        try:
            H, n, t = run_poselib(pts0, pts1, size, threshold=args.threshold, prosac=False, loops=args.loop)
            print_result("pl-lo-ransac", n, t, H is not None)
        except Exception as e:
             print(f"{'pl-lo-ransac':<20} | {'ERR':<10} | {str(e)}")

        try:
            H, n, t = run_poselib(pts0, pts1, size, threshold=args.threshold, prosac=True, loops=args.loop)
            print_result("pl-prosac", n, t, H is not None)
        except Exception as e:
             print(f"{'pl-prosac':<20} | {'ERR':<10} | {str(e)}")


    # 8. DEGENSAC
    if HAS_DEGENSAC:
        try:
            H, n, t = run_degensac(pts0, pts1, threshold=args.threshold, loops=args.loop)
            print_result("degensac", n, t, H is not None)
        except Exception as e:
             print(f"{'degensac':<20} | {'ERR':<10} | {str(e)[:30]}")

    # 9. Graph-Cut RANSAC
    if HAS_GCRANSAC:
        try:
            H, n, t = run_gcransac(pts0, pts1, size, threshold=args.threshold, affine=False, loops=args.loop)
            print_result("gcransac-std", n, t, H is not None)
            
            # Affine solver requires LAFs (Local Affine Frames) which we don't simulate here.
            # Disabling to prevent errors.
            # H, n, t = run_gcransac(pts0, pts1, size, threshold=args.threshold, affine=True, loops=args.loop)
            # print_result("gcransac-aff", n, t, H is not None)
            
        except Exception as e:
             print(f"{'gcransac':<20} | {'ERR':<10} | {str(e)[:30]}")

if __name__ == "__main__":
    main()
