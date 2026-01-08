def _status(msg: str):
    try:
        from IPython.display import clear_output
        clear_output(wait=True)
        print(msg)
    except Exception:
        # fallback for real terminals
        print("\r" + msg.ljust(80), end="", flush=True)


def _status_clear():
    try:
        from IPython.display import clear_output
        clear_output(wait=True)
    except Exception:
        print("\r" + (" " * 80), end="\r", flush=True)
