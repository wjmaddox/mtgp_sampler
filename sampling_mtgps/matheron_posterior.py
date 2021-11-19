from typing import Any, List, Optional, Union, Tuple

import torch

from botorch.posteriors import Posterior
from torch import Tensor


class MatheronPosterior(Posterior):
    def __init__(
        self,
        joint_covariance,
        test_obs_kernel,
        train_diff,
        test_mean,
        train_train_covar,
        train_noise,
    ):
        super().__init__()
        self._is_mt = True

        self.joint_covariance = joint_covariance
        self.test_obs_kernel = test_obs_kernel
        self.train_diff = train_diff
        self.test_mean = test_mean
        self.train_train_covar = train_train_covar
        self.train_noise = train_noise

        self.num_train = self.train_diff.shape[-2]
        self.num_tasks = self.test_obs_kernel.lazy_tensors[-1].shape[-1]

    @property
    def event_shape(self):
        # overwrites the standard event_shape call to inform samplers that
        # n + 2 n_train samples need to be drawn rather than n samples
        # TODO: Expose a sample shape property that is independent of the event shape
        # and handle those transparently in the samplers.
        batch_shape = self.joint_covariance.shape[:-2]
        sampling_shape = self.joint_covariance.shape[-2] + self.train_noise.shape[-2]
        return batch_shape + torch.Size((sampling_shape,))

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return self.test_mean.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return self.test_mean.dtype

    def _prepare_base_samples(
        self, sample_shape: torch.Size, base_samples: Tensor = None
    ) -> Tensor:
        covariance_matrix = self.joint_covariance
        joint_size = covariance_matrix.shape[-1]
        batch_shape = covariance_matrix.batch_shape

        if base_samples is not None:
            if base_samples.shape[: len(sample_shape)] != sample_shape:
                raise RuntimeError("sample_shape disagrees with shape of base_samples.")

            if base_samples.shape[-1] != 1:
                base_samples = base_samples.unsqueeze(-1)
            unsqueezed_dim = -2

            appended_shape = joint_size + self.train_train_covar.shape[-1]
            if appended_shape != base_samples.shape[unsqueezed_dim]:
                # get base_samples to the correct shape by expanding as sample shape,
                # batch shape, then rest of dimensions. We have to add first the sample
                # shape, then the batch shape of the model, and then finally the shape
                # of the test data points squeezed into a single dimension, accessed
                # from the test_train_covar.
                base_sample_shapes = (
                    sample_shape + batch_shape + self.test_obs_kernel.shape[-2:-1]
                )
                if base_samples.nelement() == base_sample_shapes.numel():
                    base_samples = base_samples.reshape(base_sample_shapes)

                    new_base_samples = torch.randn(
                        sample_shape
                        + batch_shape
                        + torch.Size((appended_shape - base_samples.shape[-1],))
                    )
                    base_samples = torch.cat((base_samples, new_base_samples), dim=-1)

                    base_samples = base_samples.unsqueeze(-1)
                else:
                    # nuke the base samples if we cannot use them.
                    base_samples = None

        if base_samples is None:
            # TODO: Allow qMC sampling
            base_samples = torch.randn(
                *sample_shape,
                *batch_shape,
                joint_size,
                1,
                device=covariance_matrix.device,
                dtype=covariance_matrix.dtype,
            )

            noise_base_samples = torch.randn(
                *sample_shape,
                *batch_shape,
                self.train_train_covar.shape[-1],
                1,
                device=covariance_matrix.device,
                dtype=covariance_matrix.dtype,
            )
        else:
            # finally split up the base samples
            noise_base_samples = base_samples[..., joint_size:, :]
            base_samples = base_samples[..., :joint_size, :]

        return base_samples, noise_base_samples

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
        train_diff: Optional[Tensor] = None,
    ) -> Tensor:

        if train_diff is None:
            train_diff = self.train_diff

        base_samples, noise_base_samples = self._prepare_base_samples(
            sample_shape=sample_shape, base_samples=base_samples
        )
        joint_samples = self._draw_from_base_covar(self.joint_covariance, base_samples)
        noise_samples = self._draw_from_base_covar(self.train_noise, noise_base_samples)

        # pluck out the train + test samples and add the likelihood's noise to the train side
        obs_samples = joint_samples[..., : (self.num_tasks * self.num_train)]
        updated_obs_samples = obs_samples + noise_samples
        test_samples = joint_samples[..., (self.num_tasks * self.num_train) :]

        obs_minus_samples = (
            train_diff.reshape(*train_diff.shape[:-2], -1) - updated_obs_samples
        )
        train_covar_plus_noise = self.train_train_covar + self.train_noise
        obs_solve = train_covar_plus_noise.inv_matmul(obs_minus_samples.unsqueeze(-1))

        # and multiply the test-observed matrix against the result of the solve
        updated_samples = self.test_obs_kernel.matmul(obs_solve).squeeze(-1)

        # finally, we add the conditioned samples to the prior samples
        final_samples = test_samples + updated_samples

        # and reshape
        final_samples = final_samples.reshape(
            *final_samples.shape[:-1], self.test_mean.shape[-2], self.num_tasks
        )
        final_samples = final_samples + self.test_mean
        return final_samples

    def _draw_from_base_covar(self, covar, base_samples):
        # Now reparameterize those base samples
        covar_root = covar.root_decomposition().root
        # If necessary, adjust base_samples for rank of root decomposition
        if covar_root.shape[-1] < base_samples.shape[-2]:
            base_samples = base_samples[..., : covar_root.shape[-1], :]
        elif covar_root.shape[-1] > base_samples.shape[-2]:
            raise RuntimeError("Incompatible dimension of `base_samples`")
        res = covar_root.matmul(base_samples)

        return res.squeeze(-1)
