"""Synthetic fixtures for baseline unit tests (T0, U3 Cluster A).

Two fixture cases:

Case A (delta-0): all entries share one stratum of equal size, so
bare-lambda and lambda*size rankings COINCIDE.  Hand-computed kappa is
7/11 for the full set, -1.0 for the (E1, E2) measurable-subset.

Case B (delta-nonzero): entries span strata of different sizes, so
bare-lambda and lambda*size rankings DIVERGE.  Incidence kappa is 1.0
(vote matches incidence perfectly); bare-lambda kappa is -1/11;
method_kappa_delta = -1/11 - 1 = -12/11.

All draws are constant (same lambda and vote values every draw) so the
hand-computed kappa is exact and not subject to sampling variance.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Case A: delta-0  (bare-lambda ranking == lambda*size ranking)
# ---------------------------------------------------------------------------

ENTRY_IDS_A: tuple[str, ...] = ("E1", "E2", "E3", "E4")

ENTRY_STRATA_A: dict[str, tuple[str, ...]] = {
    "E1": ("s",),
    "E2": ("s",),
    "E3": ("s",),
    "E4": ("s",),
}

STRATUM_SIZES_A: dict[str, int] = {"s": 100}

MEASURABLE_IDS_A: tuple[str, ...] = ("E1", "E2")  # E3, E4 are not measurable

# lambda[E1]=0.9, [E2]=0.7, [E3]=0.5, [E4]=0.3
# incidence = lambda * 100 -> same ordering as bare-lambda
# inc_ranks = [1, 2, 3, 4]; bare_ranks = [1, 2, 3, 4]  (delta-0)
_LAM_A = np.array([0.9, 0.7, 0.5, 0.3], dtype=np.float64)
N_DRAWS_A = 20

# Constant draws so kappa is deterministic
LAMBDA_SAMPLES_A: npt.NDArray[np.float64] = np.tile(_LAM_A, (N_DRAWS_A, 1))

# vote ranks E1=2, E2=1, E3=3, E4=4  (E1 and E2 swapped vs incidence)
_VOTE_A = np.array([2.0, 1.0, 3.0, 4.0], dtype=np.float64)
VOTE_RANK_SAMPLES_A: npt.NDArray[np.float64] = np.tile(_VOTE_A, (N_DRAWS_A, 1))

# Hand-computed kappa for Case A (full 4-entry set), tier_boundaries=(1,2):
# E1: inc=tier0(rank1), vote=tier1(rank2) -> (0,1)
# E2: inc=tier1(rank2), vote=tier0(rank1) -> (1,0)
# E3: inc=tier2(rank3), vote=tier2(rank3) -> (2,2)
# E4: inc=tier2(rank4), vote=tier2(rank4) -> (2,2)
# obs = {(0,1):0.25,(1,0):0.25,(2,2):0.5}
# obs_disagree = 1*0.25 + 1*0.25 = 0.5
# exp_disagree = 0.0625+0.5+0.0625+0.125+0.5+0.125 = 1.375
# kappa = 1 - 0.5/1.375 = 1 - 4/11 = 7/11
KAPPA_FULL_A: float = 7.0 / 11.0

# Hand-computed kappa for Case A (E1,E2 measurable-subset), tier_boundaries=(1,):
# restricted common=[E1,E2]; inc_ranks=[1,2]; vote_ranks=[2,1] (raw from vote array)
# E1: inc=tier0(1<=1), vote=tier1(2>1)  -> (0,1)
# E2: inc=tier1(2>1),  vote=tier0(1<=1) -> (1,0)
# obs_disagree = 0.5+0.5 = 1.0; exp_disagree = 0.25+0.25 = 0.5
# kappa = 1 - 1.0/0.5 = -1.0
KAPPA_SUBSET_A: float = -1.0

# ---------------------------------------------------------------------------
# Case B: delta-nonzero  (bare-lambda ranking != lambda*size ranking)
# ---------------------------------------------------------------------------

ENTRY_IDS_B: tuple[str, ...] = ("E1", "E2", "E3", "E4")

ENTRY_STRATA_B: dict[str, tuple[str, ...]] = {
    "E1": ("small", "tiny"),  # spans two strata
    "E2": ("big",),
    "E3": ("small",),
    "E4": ("tiny",),
}

STRATUM_SIZES_B: dict[str, int] = {"small": 10, "tiny": 5, "big": 200}

# lambda: E1=0.9, E2=0.1, E3=0.5, E4=0.3
# incidence: E1=0.9*(10+5)=13.5, E2=0.1*200=20, E3=0.5*10=5, E4=0.3*5=1.5
#   ordering: E2>E1>E3>E4  ->  inc_ranks=[2,1,3,4]
# bare-lambda: E1>E3>E4>E2  ->  bare_ranks=[1,4,2,3]
_LAM_B = np.array([0.9, 0.1, 0.5, 0.3], dtype=np.float64)
N_DRAWS_B = 20

LAMBDA_SAMPLES_B: npt.NDArray[np.float64] = np.tile(_LAM_B, (N_DRAWS_B, 1))

# vote ranks match incidence perfectly: [2,1,3,4]
_VOTE_B = np.array([2.0, 1.0, 3.0, 4.0], dtype=np.float64)
VOTE_RANK_SAMPLES_B: npt.NDArray[np.float64] = np.tile(_VOTE_B, (N_DRAWS_B, 1))

# Hand-computed incidence kappa for Case B (incidence vs vote perfect match):
# kappa = 1.0
KAPPA_INCIDENCE_B: float = 1.0

# Hand-computed bare-lambda kappa for Case B, tier_boundaries=(1,2):
# bare_ranks=[1,4,2,3]; vote_ranks=[2,1,3,4]
# E1: bare=tier0(1), vote=tier1(2) -> (0,1)
# E2: bare=tier2(4), vote=tier0(1) -> (2,0)
# E3: bare=tier1(2), vote=tier2(3) -> (1,2)
# E4: bare=tier2(3), vote=tier2(4) -> (2,2)
# obs_disagree = 1*0.25 + 4*0.25 + 1*0.25 = 1.5
# exp_disagree = 0.0625+0.5+0.0625+0.125+0.5+0.125 = 1.375
# bare_kappa = 1 - 1.5/1.375 = 1 - 12/11 = -1/11
KAPPA_BARE_B: float = -1.0 / 11.0

# method_kappa_delta = bare_kappa - incidence_kappa = -1/11 - 1 = -12/11
METHOD_DELTA_B: float = -12.0 / 11.0
