import torch
import speechbrain as sb


class Verifier(torch.nn.Module):
    def __init__(
        self, input_size, rnn_size, rnn_layers=2, rnn_dropout=0.1, outputs=1
    ):
        super().__init__()
        self.rnn = sb.nnet.GRU(
            hidden_size=rnn_size,
            input_size=input_size,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
        )

        self.output = sb.nnet.Linear(
            input_size=rnn_size,
            n_neurons=outputs,
            bias=False,
        )

    def forward(self, x):
        _, hidden = self.rnn(x)
        hidden = torch.add(hidden[0], hidden[1])
        return self.output(hidden)
