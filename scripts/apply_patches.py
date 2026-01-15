import os
import subprocess
import sys

def run_command(command, cwd):
    try:
        result = subprocess.run(
            command, 
            cwd=cwd, 
            shell=True, 
            text=True, 
            capture_output=True
        )
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def apply_patches():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    third_party_dir = os.path.join(base_dir, "matching", "third_party")
    patches_dir = os.path.join(base_dir, "patches")
    
    if not os.path.exists(patches_dir):
        print("Patches directory not found.")
        return

    patch_files = [f for f in os.listdir(patches_dir) if f.endswith(".patch")]
    
    if not patch_files:
        print("No patch files found.")
        return
        
    print(f"Applying patches from {patches_dir}...")
    
    success_count = 0
    fail_count = 0
    
    for patch_file in sorted(patch_files):
        submodule_name = patch_file.replace(".patch", "")
        submodule_path = os.path.join(third_party_dir, submodule_name)
        full_patch_path = os.path.join(patches_dir, patch_file)
        
        if not os.path.exists(submodule_path):
            print(f"Warning: Submodule directory {submodule_path} does not exist. Skipping {patch_file}.")
            continue
            
        print(f"Applying patch to {submodule_name}...")
        
        # Check if patch is already applied or check status?
        # git apply --check first
        stdout, stderr, rc = run_command(f"git apply --check {full_patch_path}", submodule_path)
        
        if rc != 0:
            print(f"  [!] Check failed for {submodule_name}. Patch might typically fail or already be applied.")
            print(f"      Error: {stderr.strip()}")
            # Decide whether to continue trying to apply or skip. 
            # Often 'git apply' is atomic. 
            # Let's try to apply anyway and report error if it fails? 
            # Or better, just report the failure.
            fail_count += 1
        else:
            stdout, stderr, rc = run_command(f"git apply {full_patch_path}", submodule_path)
            if rc == 0:
                print(f"  [+] Successfully applied patch to {submodule_name}")
                success_count += 1
            else:
                print(f"  [x] Failed to apply patch to {submodule_name}")
                print(f"      Error: {stderr.strip()}")
                fail_count += 1

    print("-" * 30)
    print(f"Summary: {success_count} succeeded, {fail_count} failed.")

if __name__ == "__main__":
    apply_patches()
