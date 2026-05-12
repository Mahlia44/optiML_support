from collections import OrderedDict
from typing import Optional
import torch.nn as nn

class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act_func: Optional[nn.Module] = None,
        skip_connections: bool = False
    ):
        super().__init__()
        modules = OrderedDict()
        modules["linear"] = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        modules["activation"] = act_func if act_func is not None else nn.Identity()
        self.linear_act_block = nn.Sequential(modules)
        self.skip_connections = skip_connections
        if skip_connections and in_features != out_features:
            self.adjust_dim = nn.Linear(in_features, out_features, bias=False)
        else:
            self.adjust_dim = None

    def forward(self, x):
        if self.skip_connections:
            x_adj = self.adjust_dim(x) if self.adjust_dim is not None else x
            return x_adj + self.linear_act_block(x)
        return self.linear_act_block(x)

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: list[int],
        out_features: int,
        bias: bool = True,
        act_func: Optional[nn.Module] = None,
        skip_connections: bool = False,
        init_method: str = "he_normal"
    ) -> None:
        super(MLP, self).__init__()
        assert len(hidden_features) >= 1

        modules = OrderedDict()
        hidden_dims = [in_features] + hidden_features + [out_features]

        for i in range(len(hidden_dims) - 1):
            modules[f"layer_{i}"] = LinearBlock(
                in_features=hidden_dims[i],
                out_features=hidden_dims[i + 1],
                skip_connections=skip_connections,
                # Use the activation function for all layers but last fc
                act_func=act_func if i < len(hidden_dims) - 2 else None,
                bias=bias,
            )

        self.layers = nn.Sequential(modules)
        self.init_weights(self.layers, init_method)
        self.penultimate = None

    def init_weights(self, module, init_method):
        for child in module.children():
            if isinstance(child, nn.Linear):
                self.apply_init(child, init_method)
            else:
                self.init_weights(child, init_method)

    def apply_init(self, linear_layer, init_method):
        if init_method == "xavier_uniform":
            nn.init.xavier_uniform_(linear_layer.weight)
        elif init_method == "xavier_normal":
            nn.init.xavier_normal_(linear_layer.weight)
        elif init_method == "he_normal":
            nn.init.kaiming_normal_(linear_layer.weight, nonlinearity="relu")
        elif init_method == "he_uniform":
            nn.init.kaiming_uniform_(linear_layer.weight, nonlinearity="relu")

    def embeddings(self, x):
        activations = []
        for layer in self.layers[:-1]:
            x = layer(x)
            activations.append(x)
        return activations

    def forward(self, x):
        self.penultimate = self.layers[:-1](x)
        logits = self.layers[-1](self.penultimate)
        return logits
