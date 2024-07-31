import torch.nn as nn

class IntmdSequential(nn.Sequential):
    """
    A custom nn.Sequential class that optionally returns intermediate outputs.

    Args:
        *args: Variable length argument list to pass to the nn.Sequential constructor.
        ret (bool, dict, or list): Determines how intermediate outputs are returned.
                                   If a dict, the keys should be layer names to include in the outputs.
                                   If a list, the elements will be the outputs of each layer in order.
                                   Otherwise, the forward method will return only the final output.
                                   Default is an empty dict, meaning all intermediate outputs are stored.
    """
    def __init__(self, *args, ret=dict()):
        super().__init__(*args)
        self.ret = ret

    def forward(self, input):
        if isinstance(self.ret, dict):
            output = input
            for name, module in self.named_children():
                output = self.ret[name] = module(output)
        elif isinstance(self.ret, list):
            output = input
            for name, module in self.named_children():
                output = module(output)
                self.ret.append(output)
        else:
            return super().forward(input)
        return output, self.ret
        
