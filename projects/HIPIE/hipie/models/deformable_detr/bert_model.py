from copy import deepcopy
import numpy as np
import torch
from torch import nn

# from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertConfig, RobertaConfig, RobertaModel, BertModel


class BertEncoder(nn.Module):
    def __init__(self, cfg):
        super(BertEncoder, self).__init__()
        self.cfg = cfg
        self.bert_name = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        self.max_seq_len = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
        if self.bert_name == "bert-base-uncased":
            config = BertConfig.from_pretrained("projects/HIPIE/%s"%self.bert_name)
            config.gradient_checkpointing = self.cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT
            self.model = BertModel.from_pretrained("projects/HIPIE/%s" % self.bert_name, add_pooling_layer=False, config=config)
            self.language_dim = 768
        elif self.bert_name == "roberta-base":
            config = RobertaConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = self.cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT
            self.model = RobertaModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
            self.language_dim = 768
        else:
            raise NotImplementedError

        self.num_layers = cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS # 1
        self.parallel_det = cfg.MODEL.PARALLEL_DET

    def forward(self, x, task=None,sep=None):
        input = x["input_ids"] # (bs, seq_len)
        mask = x["attention_mask"] # (bs, seq_len)

        if self.parallel_det and task == "detection":
            # disable interaction among tokens
            bs, seq_len = mask.shape
            mask_new = torch.zeros((bs, seq_len, seq_len), device=mask.device)
            for _ in range(bs):
                mask_new[_, :, :] = mask[_]
                num_valid = torch.sum(mask[_])
                mask_new[_, :num_valid, :num_valid] = torch.eye(num_valid)
                if sep is not None:
                    seps = torch.where(input[_] == sep)[0].tolist()
                    seps.insert(0,0)
                    seps.append(num_valid)
                    for i,j in zip(seps[:-1],seps[1:]):
                        start = i + 1
                        end = j 
                        mask_new[_,start:end,start:end] = 1

            # with padding, always 256
            outputs = self.model(
                input_ids=input,
                attention_mask=mask_new,
                output_hidden_states=True,
            )
        else:
            # with padding, always 256
            old_shape = input.shape
            if old_shape[1] <= 512:
                outputs = self.model(
                    input_ids=input,
                    attention_mask=mask,
                    output_hidden_states=True,
                )
            else: # need some special process
                PAD_VAL = 0
                bs, seq_len = mask.shape
                CLS = 101
                SEP = sep  # 1012
                EOS = 102
                all_inputs = []
                for bs_i in range(bs):
                    input_bs = input[bs_i]
                    mask_bs = mask[bs_i]
                    begin = 0
                    start_src = 0
                    while True:
                        seps = torch.where( (input_bs == sep)| (input_bs == EOS ))[0]
                        seps = seps[seps<510]
                        if len(seps) == 0:
                            break
                        last_sep = seps[-1]
                        first_input = input_bs[:last_sep+1]
                        first_input[-1] = EOS
                        first_mask = mask_bs[:last_sep+1]
                        first_mask_on = torch.where( first_mask == 1)[0]
                        l_valid = len(first_input)
                        indices_first_input = (start_src,start_src+l_valid,begin,begin+l_valid)
                        first_mask_out = torch.zeros(512).to(first_input)
                        if start_src == 0:
                            pad = torch.zeros((512-len(first_input))).to(first_input) + PAD_VAL
                            #pad[0] = SEP
                            first_input = torch.cat([first_input,pad],dim=0)
                            first_mask_out[first_mask_on] = 1
                            #first_mask_out[l_valid] = 1
                        elif start_src == 1:
                            pad = torch.zeros((512-len(first_input)-1)).to(first_input) + PAD_VAL
                            pad[0] = SEP
                            first_input = torch.cat([torch.tensor([CLS]).to(first_input),
                                                    first_input,pad],dim=0)
                            first_mask_out[first_mask_on+1] = 1
                            #first_mask_out[l_valid+1] = 1
                            first_mask_out[0] = 1
                        all_inputs.append((bs_i,first_input,first_mask_out,indices_first_input))
                        start_src = 1
                        input_bs = input_bs[l_valid:]
                        begin += l_valid
                inputs_actual = torch.stack([x[1] for x in all_inputs])
                masks_actual = torch.stack([x[2] for x in all_inputs])
                outputs = self.model(
                    input_ids=inputs_actual,
                    attention_mask=masks_actual,
                    output_hidden_states=True,
                )
                last_hidden_state = outputs.hidden_states[1:][-1]
                d_model = last_hidden_state.shape[-1]
                final_hidden_state = torch.zeros(bs,seq_len,d_model).float().to(input.device)
                final_mask = torch.zeros(bs,seq_len).to(input.device).float()
                recovered_inputs = torch.zeros(bs,seq_len)
                for idx, (bs_i,first_input,first_mask_out,(start_src,end_src,start_tgt,end_tgt)) in enumerate(all_inputs):
                    final_hidden_state[bs_i][start_tgt:end_tgt] = last_hidden_state[idx][start_src:end_src]
                    final_mask[bs_i][start_tgt:end_tgt] = masks_actual[idx][start_src:end_src]
                    recovered_inputs[bs_i][start_tgt:end_tgt] = inputs_actual[idx][start_src:end_src]
                # sanity check
                assert torch.all(recovered_inputs.cpu() == input.cpu()).item()
                ret = {
                        # "aggregate": aggregate,
                        # "embedded": embedded,
                        "masks": mask,
                        "hidden":final_hidden_state
                }
                return ret
                
        # outputs has 13 layers, 1 input layer and 12 hidden layers
        encoded_layers = outputs.hidden_states[1:]
        # features = None
        # features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1) # (bs, seq_len, language_dim)

        # # language embedding has shape [len(phrase), seq_len, language_dim]
        # features = features / self.num_layers

        # embedded = features * mask.unsqueeze(-1).float() # use mask to zero out invalid token features
        # aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

        ret = {
            # "aggregate": aggregate,
            # "embedded": embedded,
            "masks": mask,
            "hidden": encoded_layers[-1]
        }
        return ret
