"""Locate dataset CSVs that live in the repository's ``data/`` folder.

Notebooks call :func:`find_data_file` so they can be run from any
working directory (the repo root, a notebook subfolder, a CI
checkout). The function walks upward from a starting directory until
it finds a folder named ``data`` containing the requested filename.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["find_data_file"]


def find_data_file(filename: str, *, start: str | Path | None = None) -> Path:
    """Locate ``data/<filename>`` by walking upward from ``start``.

    Parameters
    ----------
    filename : str
        Name of the file inside the ``data/`` folder (e.g.
        ``"Crop_recommendation.csv"``).
    start : str or Path or None, optional
        Directory to start the search from. Defaults to the current
        working directory.

    Returns
    -------
    pathlib.Path
        Absolute path to the dataset.

    Raises
    ------
    FileNotFoundError
        If no ``data/<filename>`` is found in any ancestor directory.
    """
    here = Path(start) if start is not None else Path.cwd()
    here = here.resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "data" / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate data/{filename}. Place the file in the "
        f"repository's data/ folder."
    )
