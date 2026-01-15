# Notes

test with `pixi run python imm_match.py --matcher xfeat-lightglue --input assets/example_pairs/outdoor/montmartre_close.jpg assets/example_pairs/outdoor/montmartre_far.jpg --outlier cv-usac-fast`

matcher optional:

- liftfeat
- loftr
- eloftr
- se2loftr
- xoftr
- aspanformer
- matchanything-eloftr
- matchanything-roma
- matchformer
- sift-lightglue
- superpoint-lightglue
- disk-lightglue
- aliked-lightglue
- doghardnet-lightglue
- roma
- romav2
- tiny-roma
- dedode
- steerers
- affine-steerers
- dedode-kornia
- sift-nn
- orb-nn
- patch2pix
- superglue
- r2d2
- d2net
- duster
- master
- doghardnet-nn
- xfeat
- xfeat-star
- xfeat-lightglue
- dedode-lightglue
- gim-dkm
- gim-lightglue
- omniglue
- omniglue-onnx
- xfeat-subpx
- xfeat-lightglue-subpx
- dedode-subpx
- superpoint-lightglue-subpx
- aliked-lightglue-subpx
- sift-sphereglue
- superpoint-sphereglue
- minima
- minima-roma
- minima-roma-tiny
- minima-superpoint-lightglue
- minima-loftr
- ufm
- rdd
- rdd-star
- rdd-lightglue
- rdd-aliked
- minima-xoftr
- edm
- lisrd-aliked
- lisrd-superpoint
- lisrd
- lisrd-sift
- ripe
- topicfm
- topicfm-plus
- silk

failed matcher:

- romav2 oom
- duster not try
- master not try
- gim-dkm oom
- sift-sphereglue ModuleNotFoundError: No module named 'torch_geometric'
- superpoint-sphereglue ModuleNotFoundError: No module named 'torch_geometric'
- minima-roma oom

outlier method supported:

- cv-ransac
- cv-magsac
- cv-gc-ransac
- cv-usac-fast
- superansac
- pl-lo-ransac
- pl-prosac
- degensac

---

**The listed models are grouped into sparse, semi-dense, and dense categories based on their matching paradigms in computer vision literature.** Sparse methods use explicit keypoint detectors for discrete features. Semi-dense produce matches at many but not all pixels. Dense establish correspondences across nearly all pixels.

## Sparse

- liftfeat
- sift-lightglue, superpoint-lightglue, disk-lightglue, aliked-lightglue, doghardnet-lightglue, xfeat-lightglue, dedode-lightglue, gim-lightglue
- sift-nn, orb-nn, doghardnet-nn
- superglue, r2d2, d2net
- xfeat, dedode, steerers, affine-steerers, dedode-kornia
- patch2pix, omniglue, gim-dkm
- sift-sphereglue, superpoint-sphereglue
- xfeat-subpx, xfeat-lightglue-subpx, dedode-subpx, superpoint-lightglue-subpx, aliked-lightglue-subpx
- minima-superpoint-lightglue
- rdd, rdd-lightglue, rdd-aliked
- lisrd-sift, lisrd-superpoint, lisrd-aliked
- ripe
- silk

## Semi-Dense

- loftr, eloftr, se2loftr, xoftr, aspanformer, matchformer
- minima-loftr, minima-xoftr
- matchanything-eloftr
- xfeat-star, rdd-star
- topicfm, topicfm-plus
- edm

## Dense

- roma, romav2, tiny-roma
- duster, master
- minima-roma
- ufm
- matchanything-roma

## Output matcher wrapper

{
"num_inliers": int, # Jumlah inliers setelah RANSAC
"H": np.ndarray, # Matriks Homography (3x3)
"all_kpts0": np.ndarray, # Semua keypoints terdeteksi di gambar 0 (Nx2)
"all_kpts1": np.ndarray, # Semua keypoints terdeteksi di gambar 1 (Mx2)
"all_desc0": np.ndarray, # Semua deskriptor di gambar 0 (NxD)
"all_desc1": np.ndarray, # Semua deskriptor di gambar 1 (MxD)
"matched_kpts0": np.ndarray, # Keypoints di img0 yang cocok (sebelum RANSAC)
"matched_kpts1": np.ndarray, # Keypoints di img1 yang cocok (sebelum RANSAC)
"inlier_kpts0": np.ndarray, # Keypoints di img0 yang valid menurut RANSAC
"inlier_kpts1": np.ndarray # Keypoints di img1 yang valid menurut RANSAC
}

Skip matcher

- roma
- romav2
- tiny-roma
- matchanything-roma
- duster
- master
- ufm
- omniglue
- omniglue-onnx
- doghardnet-nn
- gim-dkm
- sift-sphereglue
- superpoint-sphereglue
- minima
- minima-roma
- minima-roma-tiny
- minima-superpoint-lightglue
- minima-loftr
- minima-xoftr
- topicfm
- topicfm-plus

Non Skip mathcer:

- liftfeat
- loftr
- eloftr
- se2loftr
- xoftr
- aspanformer
- matchanything-eloftr
- matchformer
- sift-lightglue
- superpoint-lightglue
- disk-lightglue
- aliked-lightglue
- doghardnet-lightglue
- dedode
- steerers
- affine-steerers
- dedode-kornia
- sift-nn
- orb-nn
- patch2pix
- superglue
- r2d2
- d2net
- xfeat
- xfeat-star
- xfeat-lightglue
- dedode-lightglue
- gim-lightglue
- xfeat-subpx
- xfeat-lightglue-subpx
- dedode-subpx
- superpoint-lightglue-subpx
- aliked-lightglue-subpx
- rdd
- rdd-star
- rdd-lightglue
- rdd-aliked
- minima-xoftr
- edm
- lisrd-aliked
- lisrd-superpoint
- lisrd
- lisrd-sift
- ripe
- silk