import torch
import torch.nn as nn

class CharCNN(nn.Module):
    def __init__(self, 
                 char_emb_dim,
                 char_input_dim,
                 char_emb_dropout,
                 char_cnn_filter_num,
                 char_cnn_kernel_size,
                 char_cnn_dropout
                ):
        super(CharCNN, self).__init__()

        # Character Embedding
        self.char_pad_idx = 0
        self.char_emb_dim = char_emb_dim
        self.char_emb = nn.Embedding(
            num_embeddings=char_input_dim,
            embedding_dim=char_emb_dim,
            padding_idx=self.char_pad_idx
        )
        
        # Initialize embedding for char padding as zero
        self.char_emb.weight.data[self.char_pad_idx] = torch.zeros(char_emb_dim )
        self.char_emb_dropout = nn.Dropout(char_emb_dropout)
        
        # Char CNN
        self.char_cnn = nn.Conv1d(
            in_channels=char_emb_dim,
            out_channels=char_emb_dim * char_cnn_filter_num,
            kernel_size=char_cnn_kernel_size,
            groups=char_emb_dim  # different 1d conv for each embedding dim
        )
        self.char_cnn_dropout = nn.Dropout(char_cnn_dropout)

    def forward(self, chars, device):
        char_emb_out = self.char_emb_dropout(self.char_emb(chars))
        batch_size, sent_len, word_len, char_emb_dim = char_emb_out.shape
        
        char_cnn_max_out = torch.zeros(batch_size, sent_len, self.char_cnn.out_channels, device=device)
        
        for sent_i in range(sent_len):
            sent_char_emb = char_emb_out[:, sent_i, :, :]
            sent_char_emb_p = sent_char_emb.permute(0, 2, 1)
            char_cnn_sent_out = self.char_cnn(sent_char_emb_p)
            char_cnn_max_out[:, sent_i, :], _ = torch.max(char_cnn_sent_out, dim=2)
        char_cnn = self.char_cnn_dropout(char_cnn_max_out)
        char_cnn_p = char_cnn

        return char_cnn_p