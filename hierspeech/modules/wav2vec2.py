import torch
import transformers


class Wav2vec2(torch.nn.Module):
    def __init__(self, layer=7, w2v='mms'):

        """we use the intermediate features of mms-300m.
           More specifically, we used the output from the 7th layer of the 24-layer transformer encoder.
        """
        super().__init__()

        if w2v == 'mms':
            self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/mms-300m")
        else:
            self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-xls-r-300m")

        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        self.wav2vec2.eval()
        self.feature_layer = layer

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (B x t)
        Returns:
            y: torch.Tensor of shape(B x C x t)
        """
        outputs = self.wav2vec2(x.squeeze(1), output_hidden_states=True)
        y = outputs.hidden_states[self.feature_layer]  # B x t x C(1024)
        y = y.permute((0, 2, 1))  # B x t x C -> B x C x t
        return y

