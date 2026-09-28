"""Separate Windows shortcut target for the NYX CPU preview checkout."""
if __name__ == '__main__':
    try:
        from experiment_console.desktop import main
        from experiment_console.preview import preview_settings_path

        main(preview_settings_path())
    except Exception as error:
        import ctypes

        ctypes.windll.user32.MessageBoxW(
            None, f'Impossible d’ouvrir NYX CPU Preview.\n\n{error}',
            'NYX CPU Preview — Démarrage impossible', 0x10 | 0x10000,
        )
        raise SystemExit(1)
