import torch
import gpytorch

from gpytorch import settings, lazify
from gpytorch.lazy import DiagLazyTensor, KroneckerProductLazyTensor, RootLazyTensor
from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag


def update_root_decomposition(kernel, fant_train_covar, fant_fant_covar, inv_root=None):
    # copied over from https://github.com/cornellius-gp/gpytorch/blob/08875a1430b8b2aefbd00baca303dd4aeb3f11b9/gpytorch/models/exact_prediction_strategies.py#L102
    # [K U; U' S] = [L 0; lower_left schur_root]
    batch_shape = fant_train_covar.shape[:-2]
    L = kernel.root_decomposition().root.evaluate()

    if inv_root is None:
        L_inverse = kernel.root_inv_decomposition().root.evaluate()
    else:
        L_inverse = inv_root.root.evaluate()

    m, n = L.shape[-2:]

    lower_left = fant_train_covar.matmul(L_inverse)
    schur = fant_fant_covar - lower_left.matmul(lower_left.transpose(-2, -1))
    schur_root = schur.root_decomposition().root.evaluate()

    # Form new root Z = [L 0; lower_left schur_root]
    num_fant = schur_root.size(-2)
    new_root = torch.zeros(
        *batch_shape, m + num_fant, n + num_fant, device=L.device, dtype=L.dtype
    )
    new_root[..., :m, :n] = L
    new_root[..., m:, : lower_left.shape[-1]] = lower_left
    new_root[..., m:, n : (n + schur_root.shape[-1])] = schur_root
    return new_root


def diagonalize(lazy_tensor):
    if lazy_tensor.shape[-1] >= settings.max_cholesky_size.value():
        qmat, tmat = lanczos_tridiag(
            lazy_tensor.matmul,
            max_iter=settings.max_root_decomposition_size.value(),
            dtype=lazy_tensor.dtype,
            device=lazy_tensor.device,
            matrix_shape=lazy_tensor.shape[-2:],
            batch_shape=lazy_tensor.batch_shape,
        )
        evals, q_t = lanczos_tridiag_to_diag(tmat)
        evecs = qmat @ q_t
        npad = evecs.shape[-2] - evecs.shape[-1]

        full_evecs = torch.cat(
            (
                evecs,
                torch.zeros(
                    *lazy_tensor.batch_shape,
                    evecs.shape[-2],
                    npad,
                    device=evecs.device,
                    dtype=evecs.dtype
                ),
            ),
            dim=-1,
        )
        full_evals = torch.cat(
            (
                evals,
                torch.zeros(
                    *lazy_tensor.batch_shape, npad, device=evecs.device, dtype=evecs.dtype
                ),
            )
        )

    else:
        evals, evecs = lazy_tensor.symeig(eigenvectors=True)

        full_evals = evals
        full_evecs = evecs

    return evals, evecs, full_evals, full_evecs
