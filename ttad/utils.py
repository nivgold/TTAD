import subprocess


def is_GPU_available():
    try:
        subprocess.check_output('nvidia-smi')
        return True
    except Exception: # this command not being found can raise quite a few different errors depending on the configuration
        return False