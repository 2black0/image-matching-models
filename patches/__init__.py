from .romav2 import patch_romav2_matcher
from .lisrd import patch_lisrd_warnings
from .duster import patch_duster_print

def apply_patches():
    """
    Applies all monkey patches for third-party libraries.
    """
    # print("Applying monkey patches...") 
    # Use quiet application to avoid noise
    patch_romav2_matcher()
    patch_lisrd_warnings()
    patch_duster_print()
    # print("Monkey patches applied.")
