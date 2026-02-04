import sys
import os

def check_environment():
    """
    Enforce running only in Google Colab or local python_embeded environment.
    """
    is_colab = False
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False

    if is_colab:
        return

    # Check for python_embeded or local .venv
    executable = sys.executable.lower()
    
    if "python_embeded" in executable or ".venv" in executable:
        return

    # Allow bypassing via env var for developers who know what they are doing
    if os.environ.get("ACESTEP_SKIP_ENV_CHECK"):
        print("Warning: Environment check skipped via ACESTEP_SKIP_ENV_CHECK", file=sys.stderr)
        return

    # Fail
    print("\n" + "!"*60, file=sys.stderr)
    print("ENVIRONMENT ERROR: ACCESS DENIED", file=sys.stderr)
    print("!"*60, file=sys.stderr)
    print("This application is configured to run ONLY in:", file=sys.stderr)
    print("  1. Google Colab", file=sys.stderr)
    print("  2. Local Windows Portable environment (using 'python_embeded')", file=sys.stderr)
    print(f"\nCurrent Environment Executable: {sys.executable}", file=sys.stderr)
    print("\nIf you are running locally, please use the provided 'run.bat' or 'run_api.bat'.", file=sys.stderr)
    print("!"*60 + "\n", file=sys.stderr)
    sys.exit(1)
