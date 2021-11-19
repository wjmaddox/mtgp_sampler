from typing import Any, List, Optional, Union, Tuple

import torch
import math

from gpytorch.kernels import ScaleKernel, MultitaskKernel, MaternKernel, RBFKernel, Kernel
from gpytorch.kernels.keops import MaternKernel as KMaternKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.constraints import GreaterThan, Interval
from botorch.posteriors import GPyTorchPosterior
from gpytorch.lazy import KroneckerProductLazyTensor, LazyTensor, lazify
from gpytorch.priors import GammaPrior, Prior
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from copy import deepcopy
from torch.nn import ModuleList
from torch import Tensor

from .matheron_mtgp import KroneckerMultiTaskGP, MatheronMultiTaskGP


class EmbeddingMultitaskKernel(MultitaskKernel):
    def __init__(
        self,
        num_tasks: int,
        data_covar_module: Kernel = None,
        task_covar_module: Kernel = None,
        task_covar_prior: Prior = None,
        context_cat_feature: Optional[Tensor] = None,
        context_emb_feature: Optional[Tensor] = None,
        embs_dim_list: Optional[List[int]] = None,
        **kwargs
    ) -> None:
        if data_covar_module is None:
            data_covar_module = ScaleKernel(
                base_kernel=MaternKernel(
                    nu=2.5, ard_num_dims=1, lengthscale_prior=GammaPrior(3.0, 6.0)
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
        super(EmbeddingMultitaskKernel, self).__init__(
            num_tasks=num_tasks,
            rank=num_tasks,
            data_covar_module=data_covar_module,
            task_covar_prior=task_covar_prior,
            **kwargs,
        )

        if context_cat_feature is None:
            context_cat_feature = torch.arange(num_tasks).long().unsqueeze(-1)
        self.context_cat_feature = context_cat_feature  # row indices = context indices
        self.context_emb_feature = context_emb_feature
        #  construct emb_dims based on categorical features
        if embs_dim_list is None:
            #  set embedding_dim = 1 for each categorical variable
            embs_dim_list = [1 for _i in range(context_cat_feature.size(1))]
        n_embs = sum(embs_dim_list)
        self.emb_dims = [
            (len(context_cat_feature[:, i].unique()), embs_dim_list[i])
            for i in range(context_cat_feature.size(1))
        ]
        # contruct embedding layer: need to handle multiple categorical features
        self.emb_layers = ModuleList(
            [
                torch.nn.Embedding(num_embeddings=x, embedding_dim=y, max_norm=1.0)
                for x, y in self.emb_dims
            ]
        )
        self.task_covar_module = RBFKernel(
            ard_num_dims=n_embs,
            lengthscale_constraint=Interval(0.0, 2.0, transform=None, initial_value=1.0),
        )

    def eval_context_covar(self) -> LazyTensor:
        """obtain context covariance matrix (num_contexts x num_contexts)"""
        all_embs = self._task_embeddings()
        return self.task_covar_module(all_embs)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError(
                "MultitaskKernel does not accept the last_dim_is_batch argument."
            )
        covar_i = self.eval_context_covar()

        if len(x1.shape[:-2]):
            covar_i = covar_i.repeat(*x1.shape[:-2], 1, 1)
        covar_x = lazify(self.data_covar_module.forward(x1, x2, **params))
        res = KroneckerProductLazyTensor(covar_x, covar_i)
        return res.diag() if diag else res

    def _task_embeddings(self) -> Tensor:
        """generate embedding features for all contexts."""
        embeddings = [
            emb_layer(self.context_cat_feature[:, i].to(dtype=torch.long))  # pyre-ignore
            for i, emb_layer in enumerate(self.emb_layers)
        ]
        embeddings = torch.cat(embeddings, dim=1)

        # add given embeddings if any
        if self.context_emb_feature is not None:
            embeddings = torch.cat(
                [embeddings, self.context_emb_feature], dim=1  # pyre-ignore
            )
        return embeddings


class KroneckerLCEMGP(KroneckerMultiTaskGP):
    r"""
    A block structured ICM multitask gp model with a Matheron posterior.
    Each task has its own latent embedding dimension as in 
    https://github.com/pytorch/botorch/blob/master/botorch/models/contextual_multioutput.py

    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: MultitaskGaussianLikelihood = None,
        task_covar_prior: Optional[Prior] = None,
        output_tasks: Optional[List[int]] = None,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        use_keops=False,
        context_cat_feature: Optional[Tensor] = None,
        context_emb_feature: Optional[Tensor] = None,
        embs_dim_list: Optional[List[int]] = None,
    ) -> None:

        super(KroneckerLCEMGP, self).__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            output_tasks=output_tasks,
            rank=None,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        kernel_fn = KMaternKernel if use_keops else MaternKernel
        ard_num_dims = train_X.shape[-1]
        batch_shape = train_X.shape[:-2]

        if context_cat_feature is None:
            context_cat_feature = (
                torch.arange(self._num_outputs).to(train_X).long().unsqueeze(-1)
            )

        self.covar_module = EmbeddingMultitaskKernel(
            data_covar_module=ScaleKernel(
                kernel_fn(
                    nu=2.5,
                    ard_num_dims=ard_num_dims,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                    batch_shape=batch_shape,
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
                batch_shape=batch_shape,
            ),
            num_tasks=self._num_outputs,
            batch_shape=batch_shape,
            context_cat_feature=context_cat_feature,
            context_emb_feature=context_emb_feature,
            embs_dim_list=embs_dim_list,
        )

        self.to(train_X)


class MatheronLCEMGP(MatheronMultiTaskGP, KroneckerLCEMGP):
    def __init__(self, *args, **kwargs):
        KroneckerLCEMGP.__init__(self, *args, **kwargs)

    def _task_covar_matrix(self) -> LazyTensor:
        return self.covar_module.eval_context_covar()
