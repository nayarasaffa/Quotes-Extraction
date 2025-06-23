import torch
from transformers import XLMRobertaTokenizer

class EntityDataset:
    def __init__(self, texts, tags, enc_tag, char_vocab):
        self.texts = texts
        self.tags = tags
        self.enc_tag=enc_tag
        self.char_vocab = char_vocab
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        MAX_LEN = 256
        CHAR_MAX_LEN = 32
        
        text = self.texts[item]
        tags = self.tags[item]

        ids = []
        char_ids = []
        target_tag =[]

        for i, s in enumerate(text):
            inputs = self.tokenizer.encode(
                str(s),
                add_special_tokens=False
            )
            tokens = self.tokenizer.tokenize(s)
            input_len = len(inputs)
            ids.extend(inputs)

            # Tag
            target_tag.extend([tags[i]] * input_len)

            # Char
            char, input_char_ids = [], []
            for token in tokens:
                if token == self.tokenizer.unk_token:
                    char.append(token)
                    input_char_ids.append([self.char_vocab.get(token, 0)])
                else:
                    character = [char for char in token]
                    character_ids = [self.char_vocab[i] for i in token]

                    char.append(character)
                    input_char_ids.append(character_ids)
            char_ids.extend(input_char_ids)
            
        ids = ids[:MAX_LEN - 2]
        char_ids = char_ids[:MAX_LEN - 2]
        target_tag = target_tag[:MAX_LEN - 2]

        # Add special token: <s> and </s>
        CLS_ID = self.tokenizer.cls_token_id
        SEP_ID = self.tokenizer.sep_token_id
        ids = [CLS_ID] + ids + [SEP_ID]

        char_ids = [[0]] + char_ids + [[0]]
        
        o_tag=self.enc_tag.transform(["O"])[0]
        target_tag = [o_tag] + target_tag + [o_tag]

        # Masking
        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids) # Not used in XLM-R, just for compatibility

        padding_len = MAX_LEN - len(ids)

        ids = ids + ([self.tokenizer.pad_token_id] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        char_ids = [i[:CHAR_MAX_LEN] for i in char_ids]
        char_ids = [i + [0] * (CHAR_MAX_LEN - len(i)) for i in char_ids] + \
            [[0] * CHAR_MAX_LEN] * (MAX_LEN - len(char_ids))

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "chars": torch.tensor(char_ids, dtype=torch.long),
        }