import torch


def get_pooling(args):
    """Helper function to return correct pooling functions."""
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
    """Mean pool for RNNs"""
    def __init__(self, args) -> None:
        super().__init__()
        self.device = args.device

    def __call__(
            self, 
            padded_prem_outp, h_n_prem, 
            padded_hypo_outp, h_n_hypo,
            premise_mask, hypothesis_mask
    ) -> torch.Tensor:
        # problems arises when complete tensor is filled with padding as 
        # we are dividing by zero. Therefore div by max(# non masks, 1)
        div_by = torch.max(
            (premise_mask == False).sum(dim=1).unsqueeze(-1), 
            torch.ones((premise_mask.shape[0], 1)).to(self.device)
        )
        premise_mean = torch.sum(padded_prem_outp, dim=1)  / div_by

        div_by = torch.max(
            (hypothesis_mask == False).sum(dim=1).unsqueeze(-1), 
            torch.ones((hypothesis_mask.shape[0], 1)).to(self.device)
        )
        hypothesis_mean = torch.sum(padded_hypo_outp, dim=1)  / div_by

        return torch.concat((premise_mean, hypothesis_mean), dim=1)


class SumPooling(torch.nn.Module):
    def __init__(self, args) -> None:
        """Sum pooling for RNNs"""
        super().__init__()

    def __call__(
            self, 
            padded_prem_outp, h_n_prem, 
            padded_hypo_outp, h_n_hypo,
            premise_mask, hypothesis_mask
    ) -> torch.Tensor:
        x = torch.concat(
            (padded_prem_outp, padded_hypo_outp),
            dim=1
        )
        x = torch.sum(x, dim=1)
        return x


class MaxPooling(torch.nn.Module):
    def __init__(self, args) -> None:
        """Max pooling for RNNs"""
        super().__init__()

    def __call__(
            self, 
            padded_prem_outp, h_n_prem, 
            padded_hypo_outp, h_n_hypo,
            premise_mask, hypothesis_mask
    ) -> torch.Tensor:
        x = torch.concat(
            (padded_prem_outp, padded_hypo_outp),
            dim=1
        )
        x = torch.max(x, dim=1)[0]
        return x


class CombinedPoolingLastHidden(torch.nn.Module):
    """Combined pooling for RNNs"""
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
                h_n_prem, h_n_hypo, # hidden state as is
                # absolute value of diff between hidden states
                torch.abs(h_n_prem - h_n_hypo), 
                # hidden states mutliplied 
                h_n_prem * h_n_hypo
            ), 
            dim = 1
        )
        return combined
