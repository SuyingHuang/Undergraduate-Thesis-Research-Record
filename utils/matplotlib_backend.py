"""Configure Matplotlib safely on desktop and headless Linux hosts."""

import os


def is_headless():
    """Return whether plots should be rendered without an interactive window."""
    override = os.environ.get('LDA_HEADLESS')
    if override is not None:
        return override.strip().lower() in {'1', 'true', 'yes', 'on'}
    return os.name != 'nt' and not (
        os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
    )


def configure_matplotlib():
    """Select the non-interactive backend before pyplot is imported."""
    import matplotlib

    if is_headless() and 'MPLBACKEND' not in os.environ:
        matplotlib.use('Agg')


def should_show_plots():
    """Interactive show is useful only with a display and GUI backend."""
    import matplotlib

    return not is_headless() and 'agg' not in matplotlib.get_backend().lower()
