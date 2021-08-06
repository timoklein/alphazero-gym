import math
from typing import List
import torch
import torch.nn.functional as F
import torch.distributions as D

__all__ = ["SquashedNormal", "GeneralizedBeta"]


class ScaledTanhTransform(D.transforms.Transform):
    """Scales a distribution between symmetric bounds.

     A more generalized version of the tanh squashing from the SAC paper.
     Based on https://arxiv.org/pdf/1801.01290.pdf.
     First squashes the samples using tanh, then rescales to (-bound, +bound).

    Parameters
    ----------
    bound : float
        Scaling factor for the transformation. Applied after tanh squashing.
    cache_size : int
        Size of the cache. Can be either 0 or 1, default is 1.
        Caching may help when nan/inf values occur.
    """

    # member annotations
    bound: float
    codomain: D.constraints.interval
    domain = D.constraints.real
    epsilon: float
    bijective: bool = True
    sign: int = +1

    def __init__(self, bound: float, epsilon: float, cache_size: int):
        super().__init__(cache_size=cache_size)

        assert bound > 0, "Scaling factor must be positive."
        self.bound = bound
        self.codomain = D.constraints.interval(-bound, bound)
        self.epsilon = epsilon

    def __eq__(self, other: object) -> bool:
        """Determines whether this distribution and other are of the same type."""
        return isinstance(other, ScaledTanhTransform)

    def __repr__(self) -> str:
        """Returns a string representation."""
        return f"Class={type(self).__name__}, Bounds={self.codomain}"

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """Tanh squashing with subsequent rescaling.

        Parameters
        ----------
        x : torch.FloatTensor
            Input Tensor. Only works for float types.

        Returns
        -------
        torch.FloatTensor
            Tensor with the scaled tanh transformation applied.
        """
        return self.bound * x.tanh()

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse of the scaled tanh squashing.

        Parameters
        ----------
        y : torch.FloatTensor
            Input Tensor. Only works for float types.

        Returns
        -------
        torch.FloatTensor
            Tensor with the inverse of the scaled tanh transformation applied.
        """
        # The epsilon parameter is needed to increase numerical stability
        # Without it, log_probs might result in NaN values if the samples
        # come from a different instance (without cache) and are at the
        # border of the atanh domain
        return torch.atanh(y / (self.bound + self.epsilon))

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the log det jacobian `log |dy/dx|` given input and output.

        More information here https://arxiv.org/pdf/1801.01290.pdf.

        Parameters
        ----------
        x : torch.FloatTensor
            Input Tensor. Only works for float types.
        y : torch.FloatTensor
            Placeholder.

        Returns
        -------
        torch.FloatTensor
            Computed result for the log determinant of the Jacobian.
        """
        # Use more stable formula
        # The term D*log(bound) corrects for the rescaling after tanh
        # Degrades to the formula from the SAC paper if bound=1.0
        # We also need to correct for the numerical stability correction in _inverse
        # by rescaling the actions with
        corr_stab = 1 + self.epsilon / self.bound
        return x.shape[-1] * math.log(self.bound) + 2.0 * (
            math.log(2.0) - corr_stab * x - F.softplus(-2.0 * corr_stab * x)
        )


class CenterScaleTransform(D.transforms.Transform):
    """Centers a Beta distribution around 0 and then rescales it.

    This transformation is taken from https://arxiv.org/pdf/1805.09613.pdf.
    Works for the beta distribution as it has a support on [0,1].
    It first scales the samples by a factor of two,
    then subtracts 1 to center around 0.
    The transformed samples are then scaled to be in [-bound, bound].

    Parameters
    ----------
    bound : float
        Scaling factor for the transformation. Applied after tanh squashing.
    cache_size : int
        Size of the cache. Can be either 0 or 1, default is 1.
        Caching may help when nan/inf values occur.
    """

    # member annotations
    bound: float
    codomain: D.constraints.interval
    domain = D.constraints.real
    epsilon: float
    bijective: bool = True
    sign: int = +1

    def __init__(self, bound: float, epsilon: float, cache_size: int):
        super().__init__(cache_size=cache_size)

        assert bound > 0, "Scaling factor must be positive."
        self.bound = bound
        self.codomain = D.constraints.interval(-bound, bound)
        self.epsilon = epsilon

    def __eq__(self, other: object) -> bool:
        """Determines whether this distribution and other are of the same type."""
        return isinstance(other, CenterScaleTransform)

    def __repr__(self) -> str:
        """Returns a string representation."""
        return f"Class={type(self).__name__}, Bounds={self.codomain}"

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """Linear transformation to enfore action bounds.

        Parameters
        ----------
        x : torch.FloatTensor
            Input Tensor. Only works for float types.

        Returns
        -------
        torch.FloatTensor
            Tensor with the scaled tanh transformation applied.
        """
        return self.bound * (2 * x - 1)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse of the scaled tanh squashing.

        Parameters
        ----------
        y : torch.FloatTensor
            Input Tensor. Only works for float types.

        Returns
        -------
        torch.FloatTensor
            Tensor with the inverse of the scaled tanh transformation applied.
        """
        # Use the epsilon parameter in the denominator for numerical stability
        return y / (2 * self.bound + self.epsilon) + 0.5

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the log det jacobian `log |dy/dx|` given input and output.

        More information here https://arxiv.org/pdf/1805.09613.pdf.

        Parameters
        ----------
        x : torch.FloatTensor
            Input Tensor. Only works for float types.
        y : torch.FloatTensor
            Placeholder.

        Returns
        -------
        torch.FloatTensor
            Computed result for the log determinant of the Jacobian.
        """
        return torch.tensor([x.shape[-1] * (math.log(2) + math.log(self.bound))])


class SquashedNormal(D.transformed_distribution.TransformedDistribution):
    """Squashed and rescaled Normal distribution.

    Squashing and rescaling consists of applying c*tanh(x) to the samples x.
    This is a more general version of the squashed normal distribution from the SAC paper.
    The original (-1,1) scaling can be recovered by setting bounds=1, although the
    generalized implementation is slightly less efficient in this case.

    Parameters
    ----------
    loc : torch.Tensor
        Location parameters of the distribution.
        The loc parameter does not correspond to the mean because it is untransformed.
    scale : torch.Tensor
        Scale parameters of the distribution.
    bounds : float
        Scaling factor after applying tanh. Results in samples scaled to (-bounds, bounds).
    """

    # member annotations
    loc: torch.Tensor
    scale: torch.Tensor
    base_dist: D.Normal
    transforms: List[ScaledTanhTransform]

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        bound: float,
        epsilon: float = 1e-06,
        cache_size: int = 1,
    ) -> None:
        self.loc = loc
        self.scale = scale

        self.base_dist = D.Normal(loc, scale)
        transforms = [
            ScaledTanhTransform(bound=bound, epsilon=epsilon, cache_size=cache_size)
        ]
        super().__init__(self.base_dist, transforms)

    def __repr__(self) -> str:
        """Returns a string representation."""
        return f"Class={type(self).__name__}, Base={str(self.base_dist)}, Mean={self.mean}, Bounds={self.transforms[0].codomain}"

    @property
    def mean(self) -> torch.FloatTensor:
        """Determine the transformed mean of the distribution.

        Returns
        -------
        torch.FloatTensor
            Transformed location parameter of the distribution.
        """
        mu = self.loc
        return self.transforms[0](mu)

    @property
    def range(self) -> D.constraints._Interval:
        """Return the support of the distribution as torch Interval.

        Will be (-c, c) since scaled tanh squashing is applied.
        Here c are the bounds for the action space.

        Returns
        -------
        D.constraints._Interval
            Interval on which the distribution is defined.
        """
        return self.transforms[0].codomain


class GeneralizedBeta(D.transformed_distribution.TransformedDistribution):
    """Centered and rescaled Beta distribution.

    Centering around 0 is done by x=2*u -1.
    Then the sample x is rescaled to be between (-bounds, bounds) with bounds*x.
    Taken from https://arxiv.org/pdf/1805.09613.pdf.

    Parameters
    ----------
    concentration1 : torch.Tensor
        First parameter of the beta distribution. Referred to as alpha.
    concentration0 : torch.Tensor
        Second parameter of the beta distribution. Referred to as beta.
    bounds : float
        Scaling factor after applying tanh. Results in samples scaled to (-bounds, bounds).
    """

    # member annotations
    concentration1: torch.Tensor
    concentration0: torch.Tensor
    base_dist: D.Beta
    transforms: List[CenterScaleTransform]

    def __init__(
        self,
        concentration1: torch.Tensor,
        concentration0: torch.Tensor,
        bound: float,
        epsilon: float = 1e-06,
        cache_size: int = 1,
    ):
        self.concentration1 = concentration1
        self.concentration0 = concentration0

        self.base_dist = D.Beta(concentration1, concentration0)
        transforms = [
            CenterScaleTransform(bound=bound, epsilon=epsilon, cache_size=cache_size)
        ]
        super().__init__(self.base_dist, transforms)

    def __repr__(self) -> str:
        """Returns a string representation."""
        return f"Class={type(self).__name__}, Base={str(self.base_dist)}, Mean={self.mean}, Bounds={self.transforms[0].codomain}"

    @property
    def mean(self) -> torch.FloatTensor:
        """Determine the transformed mean of the distribution.

        Returns
        -------
        torch.FloatTensor
            Transformed mean of the distribution.
        """
        mean = self.concentration1 / (self.concentration1 + self.concentration0)
        return self.transforms[0](mean)

    @property
    def range(self) -> D.constraints._Interval:
        """Return the support of the distribution as torch Interval.

        Will be (-c, c) since scaled tanh squashing is applied.
        Here c are the bounds for the action space.

        Returns
        -------
        D.constraints._Interval
            Interval on which the distribution is defined.
        """
        return self.transforms[0].codomain

    def entropy(self):
        return self.base_dist._dirichlet.entropy() + self.batch_shape[-1] * (
            math.log(2) + math.log(self.transforms[0].bound)
        )
