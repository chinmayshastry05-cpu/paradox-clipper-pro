"""Stage-prefixed logging so every pipeline stage is traceable."""

import logging
import sys

_CONFIGURED = False


def _configure():
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger("clipper")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(stage):
    """Return a logger named clipper.<stage> (e.g. 'download', 'analysis')."""
    _configure()
    return logging.getLogger(f"clipper.{stage}")


def set_verbose(verbose=True):
    _configure()
    logging.getLogger("clipper").setLevel(logging.DEBUG if verbose else logging.INFO)
