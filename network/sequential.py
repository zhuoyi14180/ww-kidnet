import torch.nn as nn

class CustomSequential(nn.Sequential):
    """
    Also save the intermediate output
    """
    def __init__(self, *args, flag=True):
        super().__init__(*args)
        self.flag = flag

    def forward(self, input):
        if not self.flag:
            return super().forward(input)

        res = {}
        output = input
        for name, module in self.named_children():
            output = res[name] = module(output)

        return output, res