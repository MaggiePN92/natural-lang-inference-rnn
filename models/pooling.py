import torch


def get_pooling(args):
    if args.pooling == "mean":
        return MeanPooling
    elif args.pooling == "sum":
        return SumPooling
    elif args.pooling == "max":
        return MaxPooling
    elif args.pooling == "combined":
        assert args.pooling_multiplier == 4
        return CombinedPoolingLastHidden

    else:
        print(f"Did not recognize pooling function {args.pooling}.")
        print("Possible pooling functions are mean, max and sum.")


class MeanPooling(torch.nn.Module):
    """For RNNs"""
    def __init__(self, args) -> None:
        super().__init__()
        self.device = args.device

    def __call__(self, x : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        # problems arises when complete tensor is filled with padding as 
        # we are dividing by zero. Therefore div by max(# non masks, 1)
        div_by = torch.max(
            (mask == False).sum(dim=1).unsqueeze(-1), 
            torch.ones((mask.shape[0], 1)).to(self.device)
        )
        x = torch.sum(x, dim=1)  / div_by  
        return x


class SumPooling(torch.nn.Module):
    """For RNNs"""
    def __init__(self, args) -> None:
        super().__init__()

    def __call__(self, x : torch.Tensor, *args) -> torch.Tensor:
        x = torch.sum(x, dim=1)
        return x


class MaxPooling(torch.nn.Module):
    """For RNNs"""
    def __init__(self, args) -> None:
        super().__init__()

    def __call__(self, x : torch.Tensor, *args) -> torch.Tensor:
        x = torch.max(x, dim=1)[0]
        return x


class CombinedPoolingLastHidden(torch.nn.Module):
    """For RNNs.
    
    Inspired by: 
    https://github.com/imran3180/pytorch-nli/blob/master/models/bilstm.py
    """
    def __init__(self, args) -> None:
        super().__init__()

    def __call__(
            self, 
            padded_prem_outp, h_n_prem, 
            padded_hypo_outp, h_n_hypo,
            premise_mask, hypothesis_mask
        ) -> torch.Tensor:
        B = h_n_prem.size(1)
        h_n_prem = h_n_prem.transpose(0, 1).contiguous().view(B, -1)
        h_n_hypo = h_n_hypo.transpose(0, 1).contiguous().view(B, -1)

        combined = torch.cat(
            (
                h_n_prem, h_n_hypo, 
                torch.abs(h_n_prem - h_n_hypo), 
                h_n_prem * h_n_hypo
            ), 
            dim = 1
        )
        return combined
