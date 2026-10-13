"""Construction helpers for interchangeable wetness models."""

from __future__ import annotations

from .groundwater_wetness import GroundwaterWetnessCoupler
from .sinmap_wetness import SinmapWetnessCoupler


def create_wetness_coupler(grid, model="gdp", **kwargs):
    """Create a hydrology coupler that publishes ``soil__relative_wetness``.

    Parameters
    ----------
    grid : landlab.ModelGrid
        Model grid shared with ShallowLandslider.
    model : {"gdp", "sinmap"}
        ``"gdp"`` selects transient groundwater storage and flow;
        ``"sinmap"`` selects the instantaneous equilibrium wetness index.
    **kwargs
        Passed to the selected coupler.
    """
    model = str(model).lower()
    if model == "gdp":
        return GroundwaterWetnessCoupler(grid, **kwargs)
    if model == "sinmap":
        return SinmapWetnessCoupler(grid, **kwargs)
    raise ValueError("hydrology model must be either 'gdp' or 'sinmap'")
