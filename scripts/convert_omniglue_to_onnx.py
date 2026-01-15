"""
Convert OmniGlue TensorFlow SavedModel to ONNX format

Usage:
    pixi run python scripts/convert_omniglue_to_onnx.py
"""

import subprocess
import sys
from pathlib import Path

def convert_omniglue_matcher():
    """Convert OmniGlue matcher SavedModel to ONNX"""
    
    saved_model_path = "matching/model_weights/og_export"
    output_path = "matching/model_weights/omniglue_matcher.onnx"
    
    print("="*80)
    print("Converting OmniGlue Matcher: TensorFlow SavedModel → ONNX")
    print("="*80)
    print(f"Input:  {saved_model_path}")
    print(f"Output: {output_path}")
    print()
    
    # Check if SavedModel exists
    if not Path(saved_model_path).exists():
        print(f"❌ ERROR: SavedModel not found at {saved_model_path}")
        print("Please run OmniGlue matcher once to download weights:")
        print("  pixi run python -c \"from matching import get_matcher; get_matcher('omniglue')\"")
        return False
    
    print("Converting with tf2onnx...")
    print("-"*80)
    
    try:
        # Run tf2onnx conversion
        result = subprocess.run([
            "python", "-m", "tf2onnx.convert",
            "--saved-model", saved_model_path,
            "--output", output_path,
            "--opset", "13",
            "--tag", "serve",
            "--signature_def", "serving_default",
            "--verbose"
        ], capture_output=True, text=True, check=False)
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"\n❌ Conversion failed with exit code {result.returncode}")
            return False
        
        # Verify ONNX model
        print("\n" + "="*80)
        print("Verifying ONNX model...")
        print("="*80)
        
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model is valid")
        
        # Print model info
        print("\nModel Information:")
        print(f"  ONNX opset: {onnx_model.opset_import[0].version}")
        print(f"  Inputs ({len(onnx_model.graph.input)}):")
        for inp in onnx_model.graph.input:
            print(f"    - {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
        print(f"  Outputs ({len(onnx_model.graph.output)}):")
        for out in onnx_model.graph.output:
            print(f"    - {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
        
        # Print file size
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"\n  File size: {file_size_mb:.2f} MB")
        
        print("\n" + "="*80)
        print(f"✅ SUCCESS: ONNX model saved to {output_path}")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Conversion failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = convert_omniglue_matcher()
    sys.exit(0 if success else 1)
