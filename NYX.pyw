"""NYX Windows shortcut target: no terminal or script association required."""
if __name__ == '__main__':
    try:
        from experiment_console.desktop import main
        main()
    except Exception as error:
        # Keep an actionable GUI error even if a dependency cannot be imported.
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            None, f'Impossible d’ouvrir NYX.\n\n{error}',
            'NYX — Démarrage impossible', 0x10 | 0x10000,
        )
        raise SystemExit(1)
