"""Shared CLI options for disabling cs179._native acceleration."""

from __future__ import annotations

from typing import Annotated

import typer

from reachability.envelope import REACH_SAFETY_MARGIN

NoNativeOption = Annotated[
    bool,
    typer.Option(
        "--no-native",
        help=(
            "Use Python implementations instead of cs179._native for reach envelope "
            "and retargeting (when the extension is built)."
        ),
    ),
]

ReachSafetyOption = Annotated[
    float,
    typer.Option(
        "--reach-safety",
        help=(
            "Radial workspace scale factor in (0, 1]; multiplies directional reach "
            "limits when clamping Cartesian targets (default from REACH_SAFETY_MARGIN)."
        ),
    ),
]
