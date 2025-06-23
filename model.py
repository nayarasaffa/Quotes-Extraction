import torch
import torch.nn as nn

from cnn import CharCNN
from torchcrf import CRF
from transformers import AutoModel

class EntityModel(nn.Module):
    def __init__(self, 
                 num_tag,
                 char_emb_dim=37,
                 char_input_dim=0,
                 char_emb_dropout=0.25,
                 char_cnn_filter_num=4,
                 char_cnn_kernel_size=3,
                 char_cnn_dropout=0.25,
                 input_dim=916,
                 lstm_hidden_dim=64,
                 lstm_layers=1,
                 attn_heads=4,
                 attn_dropout=0.25
                ):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag

        # XLM-RoBERTa - Word Embedding
        self.xlm_roberta = AutoModel.from_pretrained('xlm-roberta-base')

        # CNN - Character Embedding
        self.char_cnn = CharCNN(
            char_emb_dim=char_emb_dim,
            char_input_dim=char_input_dim,
            char_emb_dropout=char_emb_dropout,
            char_cnn_filter_num=char_cnn_filter_num,
            char_cnn_kernel_size=char_cnn_kernel_size,
            char_cnn_dropout=char_cnn_dropout
        )

        # BiLSTM
        self.bilstm= nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim, 
            num_layers=lstm_layers,
            bidirectional=True, 
            batch_first=True
        )

        # Multihead Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=lstm_hidden_dim * 2,
            num_heads=attn_heads,
            dropout=attn_dropout
        )
        self.attn_layer_norm = nn.LayerNorm(lstm_hidden_dim * 2)

        # CRF
        self.dropout_tag = nn.Dropout(0.3)
        self.hidden2tag_tag = nn.Linear(lstm_hidden_dim*2, self.num_tag)
        self.crf_tag = CRF(self.num_tag, batch_first=True)

    def forward(self, ids, mask, target_tag, chars, device):
        # XLM-RoBERTa - Word Embedding
        x = self.xlm_roberta(ids, attention_mask=mask)
        encoded_layers = x.last_hidden_state

        # CNN - Character Embedding
        char_cnn_p = self.char_cnn(chars, device)

        # Concat XLM-RoBERTa & CNN
        word_features = torch.cat((encoded_layers, char_cnn_p), dim=2)

        # BiLSTM
        h, _ = self.bilstm(word_features)

        ### BEGIN MODIFIED SECTION: ATTENTION ###

        h_ = h.permute(1, 0, 2)
        key_padding_mask = (mask == 0)
        attn_out, attn_weight = self.attn(h_, h_, h_, key_padding_mask=key_padding_mask)
        attn_out_ = attn_out.permute(1, 0, 2)

        h_plus_attn = h + attn_out_
        normed_h_plus_attn = self.attn_layer_norm(h_plus_attn)
        
        ### END MODIFIED SECTION: ATTENTION ###

        # CRF
        o_tag = self.dropout_tag(normed_h_plus_attn)
        tag = self.hidden2tag_tag(o_tag)        
        mask = torch.where(mask==1, True, False)
        loss_tag = - self.crf_tag(tag, target_tag, mask=mask, reduction='token_mean')
        loss=loss_tag

        return loss.unsqueeze(0)

    def encode (self, ids, mask, chars, device):
        # XLM-RoBERTa - Word Embedding
        x = self.xlm_roberta(ids, attention_mask=mask)
        encoded_layers = x.last_hidden_state

        # CNN - Character Embedding
        char_cnn_p = self.char_cnn(chars, device)

        # Concat XLM-RoBERTa & CNN
        word_features = torch.cat((encoded_layers, char_cnn_p), dim=2)

        # BiLSTM
        h, _ = self.bilstm(word_features)

        ### BEGIN MODIFIED SECTION: ATTENTION ###

        h_ = h.permute(1, 0, 2)
        key_padding_mask = (mask == 0)
        attn_out, attn_weight = self.attn(h_, h_, h_, key_padding_mask=key_padding_mask)
        attn_out_ = attn_out.permute(1, 0, 2)

        h_plus_attn = h + attn_out_
        normed_h_plus_attn = self.attn_layer_norm(h_plus_attn)
        
        ### END MODIFIED SECTION: ATTENTION ###

        # CRF
        o_tag = self.dropout_tag(normed_h_plus_attn)
        tag = self.hidden2tag_tag(o_tag)        
        mask = torch.where(mask==1, True, False)
        
        tag = self.crf_tag.decode(tag, mask=mask)

        return tag