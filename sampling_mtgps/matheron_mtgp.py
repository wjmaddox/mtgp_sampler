from typing import Any, List, Optional, Union, Tuple

import torch
import math

from botorch.models.gpytorch import GPyTorchModel
from gpytorch.kernels import MultitaskKernel, MaternKernel, IndexKernel
from gpytorch.kernels.keops import MaternKernel as KMaternKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.constraints import GreaterThan, Interval
from botorch.posteriors import GPyTorchPosterior
from gpytorch.utils.memoize import cached, pop_from_cache
from gpytorch.utils.errors import CachingError
from gpytorch.lazy import (
    BatchRepeatLazyTensor,
    RootLazyTensor,
    lazify,
    DiagLazyTensor,
    KroneckerProductLazyTensor,
)
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.models import ExactGP
from gpytorch.settings import cholesky_jitter
from gpytorch.priors import (
    GammaPrior,
    Prior,
    LKJCovariancePrior,
    HorseshoePrior,
    SmoothedBoxPrior,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from copy import deepcopy

from torch import Tensor

from .linalg import diagonalize, update_root_decomposition
from .matheron_posterior import MatheronPosterior


class KroneckerMultiTaskGP(ExactGP, GPyTorchModel):
    r"""
    This multi task GP is more like the multi task gp in 
    https://github.com/cornellius-gp/gpytorch/blob/master/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.ipynb
    rather than the multi task GP implemented in botorch.models.MultiTaskGP, due to not
    having explicit task indices.

    TODO: replace with botorch PR #637
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        # num_tasks: int,
        likelihood: MultitaskGaussianLikelihood = None,
        task_covar_prior: Optional[Prior] = None,
        output_tasks: Optional[List[int]] = None,
        rank: Optional[int] = None,
        lik_rank: Optional[int] = None,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        use_keops=False,
        eta: float = 2.0,
        has_second_noise: bool = False,
    ) -> None:

        num_tasks = train_Y.shape[-1]
        batch_shape, ard_num_dims = train_X.shape[:-2], train_X.shape[-1]

        if lik_rank is None:
            lik_rank = rank

        if input_transform is not None:
            input_transform.to(train_X)
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)

        self._validate_tensor_args(X=transformed_X, Y=train_Y)

        if likelihood is None:
            likelihood = MultitaskGaussianLikelihood(
                num_tasks=num_tasks,
                rank=lik_rank if lik_rank is not None else 0,
                noise_prior=HorseshoePrior(0.1),
                noise_constraint=Interval(1e-6, 4.0),
                has_global_noise=has_second_noise,
            )

        super(KroneckerMultiTaskGP, self).__init__(train_X, train_Y, likelihood)
        kernel_fn = KMaternKernel if use_keops else MaternKernel
        self._rank = rank if rank is not None else num_tasks
        self._num_outputs = num_tasks
        self.covar_module = MultitaskKernel(
            data_covar_module=kernel_fn(
                nu=2.5,
                ard_num_dims=ard_num_dims,
                lengthscale_prior=GammaPrior(2.0, 5.0),
                lengthscale_constraint=Interval(0.005, 100.0),
                batch_shape=batch_shape,
            ),
            num_tasks=num_tasks,
            rank=self._rank,
            batch_shape=batch_shape,
            task_covar_prior=LKJCovariancePrior(
                n=num_tasks,
                eta=eta,
                sd_prior=SmoothedBoxPrior(math.exp(-6), math.exp(1.25), 0.05),
            ),
        )

        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=num_tasks)

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.to(train_X)

    def forward(self, x: Tensor) -> MultitaskMultivariateNormal:
        x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x).evaluate_kernel()
        return MultitaskMultivariateNormal(mean_x, covar_x)


class MatheronMultiTaskGP(KroneckerMultiTaskGP):
    r"""
        A multi task gp class with a Matheron's rule sampled posterior instead of the 
        standard multi task posterior.
    """

    def _task_covar_matrix(self):
        return self.covar_module.task_covar_module.covar_matrix

    @property
    @cached(name="task_root")
    def task_root(self):
        task_covar = self._task_covar_matrix().detach()

        # construct MM' \approx Ktt
        task_evals, task_evecs, _, _ = diagonalize(task_covar)
        task_root = task_evecs.matmul(task_evals.clamp(1e-7).diag_embed().sqrt())
        return task_root

    @property
    @cached(name="train_full_covar")
    def train_full_covar(self):
        train_x = self.transform_inputs(self.train_inputs[0])

        # construct Kxx \otimes Ktt
        train_full_covar = self.covar_module(train_x).evaluate_kernel().detach()
        return train_full_covar

    @property
    @cached(name="data_data_roots")
    def data_data_roots(self):

        # construct Kxx \otimes Ktt
        train_full_covar = self.train_full_covar

        # construct Kxx
        data_data_covar = train_full_covar.lazy_tensors[0]

        # construct RR' \approx Kxx
        dd_evals, dd_evecs, _, _ = diagonalize(data_data_covar)
        q_t_root = dd_evecs.matmul(dd_evals.diag_embed().clamp(1e-7).sqrt())
        q_t_inv_root = dd_evecs.matmul((1.0 / dd_evals.clamp(1e-7).sqrt()).diag_embed())
        return q_t_root, q_t_inv_root

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        self.eval()

        X = self.transform_inputs(X)
        train_x = self.transform_inputs(self.train_inputs[0])

        ## construct Ktt
        task_covar = self._task_covar_matrix().detach()
        task_root = self.task_root
        if task_covar.batch_shape != X.shape[:-2]:
            task_covar = BatchRepeatLazyTensor(task_covar, batch_repeat=X.shape[:-2])
            task_root = BatchRepeatLazyTensor(
                lazify(task_root), batch_repeat=X.shape[:-2]
            )

        task_covar_rootlt = RootLazyTensor(task_root)

        ## construct RR' \approx Kxx
        q_t_root, q_t_inv_root = self.data_data_roots
        data_data_root = RootLazyTensor(q_t_root)
        data_data_inv_root = RootLazyTensor(q_t_inv_root)

        # construct K_{xt, x}
        test_data_covar = self.covar_module.data_covar_module(X, train_x)
        # construct K_{xt, xt}
        test_test_covar = self.covar_module.data_covar_module(X)

        # now update root so that \tilde{R}\tilde{R}' \approx K_{(x,xt), (x,xt)}
        # cloning preserves the gradient history or something
        with cholesky_jitter(1e-7):
            updated_root = update_root_decomposition(
                data_data_root,
                test_data_covar.clone(),
                test_test_covar,
                inv_root=data_data_inv_root,
            )

        # build a root decomposition of the joint train/test covariance matrix
        # construct (\tilde{R} \otimes M)(\tilde{R} \otimes M)' \approx (K_{(x,xt), (x,xt)} \otimes Ktt)
        joint_covar = RootLazyTensor(
            KroneckerProductLazyTensor(updated_root, task_covar_rootlt.root.detach())
        )

        # construct K_{xt, x} \otimes Ktt
        test_obs_kernel = KroneckerProductLazyTensor(test_data_covar, task_covar)

        # collect y - \mu(x) and \mu(X)
        train_diff = self.train_targets - self.mean_module(train_x)
        test_mean = self.mean_module(X)

        train_noise = self.likelihood._shaped_noise_covar(train_x.shape).detach()

        posterior = MatheronPosterior(
            joint_covariance=joint_covar,
            test_obs_kernel=test_obs_kernel,
            train_diff=train_diff.detach(),
            test_mean=test_mean,
            train_train_covar=self.train_full_covar,
            train_noise=train_noise,
        )

        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)

        return posterior

    def train(self, val=True, *args, **kwargs):
        if val:
            fixed_cache_names = ["data_data_roots", "train_full_covar", "task_root"]
            for name in fixed_cache_names:
                try:
                    pop_from_cache(self, name)
                except CachingError:
                    pass

        return super().train(val, *args, **kwargs)
