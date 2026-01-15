import os
import subprocess

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

def generate_patches():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    third_party_dir = os.path.join(base_dir, "matching", "third_party")
    patches_dir = os.path.join(base_dir, "patches")
    
    os.makedirs(patches_dir, exist_ok=True)
    
    print(f"Scanning for changes in {third_party_dir}...")
    
    subdirs = [d for d in os.listdir(third_party_dir) if os.path.isdir(os.path.join(third_party_dir, d))]
    
    for subdir in sorted(subdirs):
        submodule_path = os.path.join(third_party_dir, subdir)
        
        # Check if it is a git repo
        if not os.path.exists(os.path.join(submodule_path, ".git")):
            continue
            
        print(f"Checking {subdir}...")
        
        
        # Add intent to add for untracked files to make them visible to git diff, but exclude pycache
        # Using git ls-files to respect .gitignore and manually excluding common cache files if needed
        # But simply adding a local .gitignore or using exclude arguments might be easier
        # Let's try adding specific ignores to the add command or just adding everything EXCEPT pycache
        
        
        # Strategy: find all files, filter out pycache, then add them
        # Clean potential pycache garbage first
        run_command("find . -name '__pycache__' -type d -exec rm -rf {} +", submodule_path)
        run_command("find . -name '*.pyc' -type f -delete", submodule_path)

        # To be absolutely safe, let's create/append to a temp .gitignore if it doesn't exclude pycache?
        # A simpler way is to just use standard git add but we want to discern untracked files.
        # Let's rely on 'git ls-files --others --exclude-standard' if we trust the repo's gitignore,
        # but we can't assume submodules ignore pycache.
        
        
        # New approach:
        # 1. Reset everything to be safe
        run_command("git reset", submodule_path)
        
        # 2. Add all files EXCEPT:
        # - pycache/pyc
        # - binary weights (pth, pt, chenkpoint, onnx, bin, safetensors, h5, tflite, pb)
        # - git internal files
        
        # Define extensions to exclude
        exclude_exts = ['*.pyc', '*.pth', '*.pt', '*.ckpt', '*.onnx', '*.bin', '*.safetensors', '*.h5', '*.tflite', '*.pb']
        
        # Construct find command exclusions
        # -not -name '*.pyc' -not -name '*.pth' ...
        find_excludes = " ".join([f"-not -name '{ext}'" for ext in exclude_exts])
        
        find_cmd = f"find . -type f -not -path '*/__pycache__/*' {find_excludes} -not -path '*/.git/*'"
        
        # We use xargs to handle large file lists
        add_cmd = f"{find_cmd} | xargs -r git add -N"
        run_command(add_cmd, submodule_path)
        
        # 3. Generate diff, explicitly excluding patterns again just in case git diff picks them up
        # Construct pathspec exclusions for git diff
        # ':(exclude)*.pyc' ':(exclude)*.pth' ...
        diff_excludes = " ".join([f"':(exclude){ext}'" for ext in exclude_exts])
        diff_cmd = f"git diff -- . ':(exclude)*/__pycache__/*' {diff_excludes}"
        
        diff_output, stderr, _ = run_command(diff_cmd, submodule_path)
        
        # 4. Reset to leave the repo state as it was
        run_command("git reset", submodule_path)
        
        if diff_output.strip():
            patch_file = os.path.join(patches_dir, f"{subdir}.patch")
            with open(patch_file, "w") as f:
                f.write(diff_output)
            print(f"  [+] Created patch: patches/{subdir}.patch")
        else:
            print(f"  [ ] No changes in {subdir}")

if __name__ == "__main__":
    generate_patches()
