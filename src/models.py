import torch


class PARAFAC(torch.nn.Module):
    def __init__(self, dims, k, scale=1.0, bias=0.0):
        super().__init__()

        self.k = k
        self.n_factors = len(dims)
        self.factors = torch.nn.ParameterList([
            torch.nn.Parameter(scale * (torch.randn(dim, k, dtype=torch.double, requires_grad=True) - bias))
            for dim in dims
        ])

    def forward(self, indices):
        factor_vectors = []
        for i in range(indices.shape[1]):
            idx = indices[:, i]
            factor_vectors.append(self.factors[i][idx, :])        
        vectors = torch.stack(factor_vectors, dim=1)
        prod = torch.prod(vectors, dim=1)

        n_non_used_factors = self.n_factors - indices.shape[1]
        if n_non_used_factors > 0:
            res = []
            for cols in zip(
                *[self.factors[-(a + 1)].t() for a in reversed(range(n_non_used_factors))]
            ):
                kr = cols[0]
                for j in range(1, n_non_used_factors):
                    kr = torch.kron(kr, cols[j])
                res.append(kr)
            factors_action = torch.stack(res, dim=1)
            return torch.matmul(prod, factors_action.T)

        return torch.sum(prod, dim=-1)
