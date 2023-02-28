import torch

class GPT2Dataset(Dataset):
    # Inherits Dataset from PyTorch, a data primitive which
    # stores samples and corresponding labels
    # custom Dataset needs init, len, and getitem
    # init runs once when instantiating Dataset object
    def __init__(self, txt_list, tokenizer, gpt2_type='gpt2', max_length=768):
        self.input_ids = []
        self.attn_masks = []
        
        # for each text list, encode it using tokenizer then unpacl encodings dict into:
        # input_ids: numerical representations of our tokens
        # attn_masks: indicates which tokens should be attended to (and which are pads)
        for txt in txt_list:
            # tokenize the txt with a custom start and end token
            # encodings dict contains both our token input ids and attention mask
            # truncation will clip sentences that are too long
            # padding adds pad tokens until we reach max input sentence length 768
            encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
            
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    # overrides len() to returns the number of samples in our dataset
    def __len__(self):
        return len(self.input_ids)
    # loads and returns a sample from dataset at given index idx
    # sometimes we need to do type swapping in getitem, but not here
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]