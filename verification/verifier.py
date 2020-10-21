import torch
import speechbrain as sb


class Verifier(torch.nn.Module):
    def __init__(
        self,
        input_size,
        vocab_size,
        emb_size=32,
        rnn_size=256,
        rnn_layers=2,
        rnn_dropout=0.1,
        char_rnn_size=256,
        char_rnn_layers=2,
        char_rnn_dropout=0.1,
        dnn_size=256,
        dnn_layers=2,
        dnn_dropout=0.1,
        outputs=1,
    ):
        super().__init__()
        #self.rnn = sb.nnet.GRU(
        #    hidden_size=rnn_size,
        #    input_size=input_size,
        #    num_layers=rnn_layers,
        #    dropout=rnn_dropout,
        #)

        self.char_embedding = sb.nnet.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=input_size,
        )
        #self.char_rnn = sb.nnet.GRU(
        #    hidden_size=char_rnn_size,
        #    input_size=emb_size,
        #    num_layers=char_rnn_layers,
        #    dropout=char_rnn_dropout,
        #)
        self.transformer = sb.lobes.TransformerDecoder(
            num_layers=8,
            nhead=8,
            d_ffn=256,
            d_model=input_size,
            dropout=0.1,
        )

        # DNN aggregator
        self.dnn_layers = torch.nn.ModuleList()
        for i in range(dnn_layers):
            self.dnn_layers.append(
                sb.nnet.Linear(input_size=input_size, n_neurons=dnn_size)
            )
            self.dnn_layers.append(sb.nnet.BatchNorm1d(input_size=dnn_size))
            self.dnn_layers.append(torch.nn.LeakyReLU())
            self.dnn_layers.append(torch.nn.Dropout(p=dnn_dropout))

            # update input size for next layer
            input_size = dnn_size

        # Final classifier
        self.output = sb.nnet.Linear(
            input_size=input_size,
            n_neurons=outputs,
            bias=False,
        )

    def forward(self, x, chars):
        """Computes verification forward pass.

        1. RNN over ASR model hidden states (take hidden state)
        2. RNN over characters being read (take hidden state)
        3. DNN aggregator over both inputs
        4. Output layer
        """

        # Compute RNN over ASR hidden states
        #_, hidden = self.rnn(x)
        #hidden = torch.add(hidden[0], hidden[1])

        # Compute RNN over characters being read
        #embedded_chars = self.char_embedding(chars)
        #_, char_hidden = self.char_rnn(embedded_chars)
        #char_hidden = torch.add(char_hidden[0], char_hidden[1])

        # Embed characters
        embedded_chars = self.char_embedding(chars)
        output, _, _ = self.transformer(embedded_chars, x)

        # Max pooling across transformer outputs
        x, _ = output.max(dim=1)

        for layer in self.dnn_layers:
            x = layer(x)

        return self.output(x)
