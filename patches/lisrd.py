import warnings

def patch_lisrd_warnings():
    """
    Suppresses warnings associated with LISRD's use of default grid_sample behavior.
    """
    # LISRD triggers: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False...
    warnings.filterwarnings("ignore", message="Default grid_sample and affine_grid behavior has changed", category=UserWarning)
    print("Monkey-patched LISRD: Suppressed grid_sample warnings.")
