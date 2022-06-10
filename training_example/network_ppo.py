class DenseNet(torch.nn.Module):
    
    def __init__(self, in_size: int, out_size: int, hidden: int = 128):
        """ Dense network that contains the value and policy functions.

        Args:
            in_size (int): Input size (length of the state vector)
            out_size (int): Action size (number of categories)
            hidden (int, optional): Hidden neuron size. Defaults to 128.
        """
        super().__init__()
        self.base = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden),
            torch.nn.ReLU()
        )

        self.mu = torch.nn.Sequential(
            torch.nn.Linear(hidden, out_size),
            torch.nn.Tanh()
        )

        self.var = torch.nn.Sequential(
            torch.nn.Linear(hidden, out_size),
            torch.nn.Softplus()
        )

        self.value = torch.nn.Linear(hidden,1)

    def forward(self, state: torch.Tensor,
                ) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        """ Return policy distribution and value

        Args:
            state (torch.Tensor): State tensor

        Returns:
            Tuple[torch.distributions.Normal, torch.Tensor]: Normal 
                policy distribution and value
        """

        x = self.base(state)
        dist = Normal(self.mu(x), self.var(x))
        return dist, self.value(x)