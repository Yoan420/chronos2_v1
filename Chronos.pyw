"""Compatibility entry for shortcuts installed before the NYX app name."""
if __name__ == '__main__':
    from pathlib import Path
    from runpy import run_path
    run_path(str(Path(__file__).with_name('NYX.pyw')), run_name='__main__')
