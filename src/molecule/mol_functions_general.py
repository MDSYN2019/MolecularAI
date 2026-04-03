import torch

def normalize_adjacency(A):
    """
    Returns the symmetrically normalized adjacency matrix

    This normalizes the adjacency matrix by accounting for node degrees, helping
    prevent bias from high-degree nodes. It uses the common normalization:

    A_hat = D^(-1/2)(A + I) D^(-1/2)

    Where A is the adjacency matrix, I is the identity, and D is thr degree matrix
    """
    I = torch.eye(A.size(0))  # Get the identity matrix
    A_hat = A + I  # add the identity matrix to the adjacency matrix
    D_hat = torch.diag(torch.pow(A_hat.sum(1), -0.5))  # return the normalized adjacency
    return D_hat @ A_hat @ D_hat

def align_targets(
    pred: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ensure y and y_mask match pred shape [B, T].
    Accepts y / mask as [B], [B*T], or already [B, T].
    """
    B, T = pred.shape

    # --- fix y ---
    if y is None:
        raise RuntimeError("Target y is None")
    if y.dim() == 1:  # [B] or [B*T]
        if y.numel() == B * T:
            y = y.view(B, T)
        elif y.numel() == B:
            y = y.view(B, 1).expand(B, T)
        else:
            raise RuntimeError(
                f"Cannot reshape y of shape {tuple(y.shape)} to [B,T]=[{B},{T}]"
            )
    elif y.dim() == 2 and (y.shape[0] == B and y.shape[1] == T):
        pass  # good
    else:
        # last resort: same numel?
        if y.numel() == B * T:
            y = y.view(B, T)
        else:
            raise RuntimeError(
                f"Incompatible y shape {tuple(y.shape)} for pred {tuple(pred.shape)}"
            )

    # --- fix mask ---
    if y_mask is None:
        y_mask = torch.ones_like(y, dtype=y.dtype, device=y.device)
    else:
        if y_mask.dim() == 1:  # [B] or [B*T]
            if y_mask.numel() == B * T:
                y_mask = y_mask.view(B, T)
            elif y_mask.numel() == B:
                y_mask = y_mask.view(B, 1).expand(B, T)
            else:
                raise RuntimeError(
                    f"Cannot reshape mask {tuple(y_mask.shape)} to [B,T]=[{B},{T}]"
                )
        elif y_mask.dim() == 2 and (y_mask.shape[0] == B and y_mask.shape[1] == T):
            pass
        else:
            if y_mask.numel() == B * T:
                y_mask = y_mask.view(B, T)
            else:
                raise RuntimeError(
                    f"Incompatible mask shape {tuple(y_mask.shape)} for pred {tuple(pred.shape)}"
                )

        y_mask = y_mask.to(y.device, dtype=y.dtype)

    return y, y_mask
