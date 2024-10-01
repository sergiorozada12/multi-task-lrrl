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

        if indices.shape[1] < self.n_factors:
            return torch.matmul(prod, self.factors[-1].T)

        return torch.sum(prod, dim=-1)
