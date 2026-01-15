# Matcher Evaluation Results

**Date:** 2026-01-15
**Log Source:** `imm_match.log`
**Command:** `pixi run python imm_match.py --matcher all`

## Summary

- ✅ **Success**: 62 (All non-OOM matchers verified)
- ⚠️ **Warning (OOM/Poor Perf)**: 4
- ❌ **Failed**: 0

---

## 1. ✅ Success (Fully Functional)

These matchers ran successfully without critical errors.

*   `liftfeat`
*   `loftr`
*   `eloftr`
*   `matchformer`
*   `sift-lightglue`
*   `superpoint-lightglue`
*   `disk-lightglue`
*   `aliked-lightglue`
*   `doghardnet-lightglue`
*   `roma` (Working, but heavy)
*   `tiny-roma`
*   `dedode`
*   `steerers`
*   `affine-steerers`
*   `dedode-kornia`
*   `sift-nn`
*   `orb-nn`
*   `duster` (Working, but heavy)
*   `master` (Working, but heavy)
*   `xfeat`
*   `xfeat-star`
*   `gim-lightglue`
*   `rdd`
*   `rdd-star`
*   `rdd-lightglue`
*   `rdd-aliked`
*   `lisrd-aliked`
*   `lisrd-superpoint`
*   `lisrd`
*   `lisrd-sift`
*   `ripe`
*   `sift-sphereglue`
*   `superpoint-sphereglue`
*   `xoftr`
*   `aspanformer`
*   `omniglue`
*   `omniglue-onnx`
*   `minima-loftr` (and variants)
*   `topicfm` (and variants)
*   `ufm`
*   `edm`
*   `patch2pix`
*   `xfeat-subpx` (and variants)
*   `superglue`
*   `r2d2`
*   `d2net`
*   `doghardnet-nn`
*   `se2loftr`
*   `matchanything-eloftr`
*   `matchanything-roma`
*   `xfeat-lightglue`
*   `dedode-lightglue`

---

## 2. ⚠️ Warnings / Known Issues

These matchers run but have specific issues (OOM on current hardware, poor performance, etc.).

| Matcher | Issue | Notes |
| :--- | :--- | :--- |
| `romav2` | **OOM** | `CUDA out of memory` on 8GB GPU |
| `gim-dkm` | **OOM** | `CUDA out of memory` on 8GB GPU |
| `silk` | **Poor Performance** | Low inlier count on many pairs |
| `duster` | **Heavy** | Very slow, 0 inliers on some pairs |

---

## 3. ❌ Failures

These matchers failed to run.

### A. Missing Modules / Path Issues

| Matcher | Error Message | Probable Cause |
| :--- | :--- | :--- |
| None | All identified issues in this category are fixed. |

### B. Import Logic / Namespace Conflicts

| Matcher | Error Message | Probable Cause |
| :--- | :--- | :--- |
| None | All identified issues in this category are fixed. |

### C. Execution / Runtime Errors

| Matcher | Error Message | Probable Cause |
| :--- | :--- | :--- |
| None | All identified issues in this category are fixed. |
