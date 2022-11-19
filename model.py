import copy
import math
import torch
import torch.nn as nn


def clones(module, N):
    """
        Produce N identical layers
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """
        Produce N identical layers
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """
        Vocabulary embedding function
            d_model: dimension of transformer
            vocab: number of vocabulary
        """
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab, d_model)

    def forward(self, x):
        """
            x: (batch, vocab)
            embed: embed: (batch, vocab, d_model)
        """
        embed = self.embed(x) * math.sqrt(self.d_model)
        return embed


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=5000):
        """
        Positional encoding of embedding vector
            d_model: dimension of transformer
            dropout: dropout rate
            max_len: max distance position
        """
        super(PositionalEncoding, self).__init__()
        # create positional encoding matrix: (1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        self.pe = pe.unsqueeze(0)
        # dropout for converge
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # add positional vector into input feature
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.):
        """
        Feed forward block: project low-dimension space into high-dimension space
            d_model: dimension of model
            d_hide: hidden layer of model
            dropout: dropout rate
        """
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(d_model, d_hidden)
        self.l2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.l1(x)  # (batch, vocab, d_model) => (batch, vocab, d_hidden)
        x = self.dropout(x.relu())
        x = self.l2(x)  # (batch, vocab, d_hidden) => (batch, vocab, d_model)
        return x


class ConnectNorm(nn.Module):
    def __init__(self, d_model, dropout=0.):
        """
        Add & Norm block: residual block of model
            d_model: dimension of transformer
            dropout: dropout rate
        """
        super(ConnectNorm, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layer):
        """
        Normalization & Residual
        params:
            x: (batch, vocab, d_model)
            layer: layer function
        return:
            x+r: (batch, vocab, d_model)
        """
        r = self.norm(x)
        r = self.dropout(layer(r))
        return x + r


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        """
        Scaled Dot Product Attention
        params:
            query: (batch, h, vocab, d_k)
            key: (batch, h, vocab, d_k)
            value: (batch, h, vocab, d_k)
            mask: (batch, 1, 1, vocab)
            dropout: float
        return:
            value: (batch, h, vocab, d_k)
            atten: (batch, h, vocab, vocab)
        """
        # 1) Q K multiple & scale: (batch, h, vocab, vocab)
        scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            # 2) masked attention: mask=>(batch, 1, 1, vocab)
            scores = scores.masked_fill(mask == 0, -1e9) # masked attention score
        # 3) attention score normalization: (batch, h, vocab, vocab)
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            # 4) dropout for converge
            p_attn = dropout(p_attn)
        # 5) attention at V: (batch, h, vocab, d_k)
        out = torch.matmul(p_attn, value)
        return out, p_attn

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        if mask is not None:
            # Same mask applied to all h heads: (batch, 1, vocab) => (batch, 1, 1, vacab)
            mask = mask.unsqueeze(1)
        # 1) linear projection in batch: (batch, vocab, d_model) => (batch, h, vocab, d_k)
        query, key, value = [
            linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]
        # 2) attention in batch: # (batch, h, vocab, d_k), # (batch, h, vocab, vocab)
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) concat using view: # (batch, h, vocab, d_k) => (batch, vocab, d_model)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h*self.d_k)
        # 4) linear projection
        x = self.linears[-1](x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_hide, dropout):
        """
        Encoder layer of transformer
        params:
            d_model: dimension of model
            h: multi-head number
            d_hide: dimension of hidden layer
            dropout: dropout rate
        """
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        # multi-head attention layer
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        # feed forward layer
        self.feed_forward = FeedForward(d_model, d_hide)
        # short connect layer
        self.add_norm = clones(ConnectNorm(d_model, dropout), 2)

    def forward(self, x, mask):
        # encoder self attention
        x = self.add_norm[0](x, lambda x: self.self_attn(x, x, x, mask))
        # encoder feed forward
        x = self.add_norm[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, encoder_layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(encoder_layer, N)
        self.norm = LayerNorm(encoder_layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_hide, dropout):
        """
        Decoder layer of transformer
            d_model: dimension of model
            h: multi-head number
            d_hide: dimension of hidden layer
            dropout: dropout rate
        """
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        # masked multi-head attention layer
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        # multi-head attention layer
        self.src_attn = MultiHeadAttention(d_model, h, dropout)
        # feed forward layer
        self.feed_forward = FeedForward(d_model, d_hide)
        # short connect layer
        self.add_norm = clones(ConnectNorm(d_model, dropout), 3)

    def forward(self, x, m, src_mask, tgt_mask):
        x = self.add_norm[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.add_norm[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        x = self.add_norm[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, decoder_layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(decoder_layer, N)
        self.norm = LayerNorm(decoder_layer.d_model)

    def forward(self, x, m, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, m, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        """
        Standard linear + softmax generation
            d_model: dimension of transformer
            vocab: number of vocabulary
        """
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.project(x)
        x = self.softmax(x)
        return x


class Transformer(nn.Module):
    """
    A standard Encoder-Decoder architecture
    """
    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_hide=2048, h=8, dropout=0.1):
        super(Transformer, self).__init__()
        c = copy.deepcopy
        self.src_embed = Embeddings(d_model, src_vocab)
        self.tgt_embed = Embeddings(d_model, tgt_vocab)
        encoder_layer = EncoderLayer(d_model, h, d_hide, dropout)
        self.encoder = Encoder(c(encoder_layer), N)
        decoder_layer = DecoderLayer(d_model, h, d_hide, dropout)
        self.decoder = Decoder(c(decoder_layer), N)
        self.generator = Generator(d_model, tgt_vocab)

    def _init_weights(self, model):
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def encode(self, src, src_mask):
        x = self.encoder(self.src_embed(src), src_mask)
        return x

    def decode(self, memory, src_mask, tgt, tgt_mask):
        x = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask):
        feature = self.encode(src, src_mask)
        return self.decode(feature, src_mask, tgt, tgt_mask)


if __name__ == '__main__':
    test_model = Transformer(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) # (batch, vocab)
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)
