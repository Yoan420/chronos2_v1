"""NYX Windows shortcut target: no terminal or script association required."""
if __name__ == '__main__':
    try:
        from experiment_console.desktop import main
        import argparse
        parser = argparse.ArgumentParser(description='NYX desktop')
        parser.add_argument('--settings', help='Optional isolated desktop settings JSON')
        args = parser.parse_args()
        main(args.settings)
    except Exception as error:
        # Keep an actionable GUI error even if a dependency cannot be imported.
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            None, f'Impossible d’ouvrir NYX.\n\n{error}',
            'NYX — Démarrage impossible', 0x10 | 0x10000,
        )
        raise SystemExit(1)
