import argparse
import os
import torch
import numpy as np
import json
from transformers import GPT2Config, GPT2Tokenizer

import tqdm
from torch.utils.data import DataLoader
import clip
from clipscore.eval_clip import computer_clipscore_and_other
from modeling_gpt2 import GPT2LMHeadModel
from torch.autograd import Variable


from lib.config import cfg, cfg_from_file
import lib.utils as utils
from lib.dataset import RCStyleDataset_infer

from models.pure_transformer import PureT
from utils import ClipCaptionModel, eval_ppl, eval_acc



class vocab_gpt2puret():
    def __init__(self, gpt_decoder, gpt_encoder, pure_decoder):    
        th = gpt_decoder[294]
        th0 = th[0]
        idx_aban = []
        idx_value = [[], []]
        for i in range(len(pure_decoder)):
            x = pure_decoder[i]
            if x in gpt_encoder and th0+x in gpt_encoder:
                idx_value[0].append(gpt_encoder[x])
                idx_value[1].append(gpt_encoder[th0+x])
            elif x in gpt_encoder:
                idx_value[0].append(gpt_encoder[x])
                idx_value[1].append(gpt_encoder[x])
            elif th0+x in gpt_encoder:
                idx_value[0].append(gpt_encoder[th0+x])
                idx_value[1].append(gpt_encoder[th0+x])
            else:
                idx_aban.append(i)
                idx_value[0].append(0)
                idx_value[1].append(0)

        self.idx_aban = idx_aban
        self.idx_value = idx_value

    def logP_gpt2puret(self, logP_gpt):
        logP_pureT0 = logP_gpt[:, self.idx_value[0]]
        logP_pureT1 = logP_gpt[:, self.idx_value[1]]
        logP_pureT = torch.where(logP_pureT0 >= logP_pureT1, logP_pureT0, logP_pureT1)
        logP_pureT[:, self.idx_aban] -= 10000
        return logP_pureT
    
    def token_puret2gpt(self, token_puret, first):
        token_puret = token_puret.tolist()
        if first:
            token_gpt = [self.idx_value[0][idx] for idx in token_puret]
        else:
            token_gpt = [self.idx_value[1][idx] for idx in token_puret]
        return torch.tensor(token_gpt)


def decode_generate(model, kwargs, gedi_model, kwargs_gedi):

    greedy_decode = kwargs['GREEDY_DECODE']

    # image feature
    att_feats = kwargs[cfg.PARAM.ATT_FEATS]
    att_feats = model.backbone(att_feats)
    att_feats = model.att_embed(att_feats)
    gx, encoder_out = model.encoder(att_feats, None)

    # base
    batch_size = att_feats.size(0)
    model.decoder.init_buffer(batch_size)
    state = None
    sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).to(att_feats.device))
    logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).to(att_feats.device))
    wt = Variable(torch.zeros(batch_size, dtype=torch.long).to(att_feats.device))
    unfinished = wt.eq(wt)
    kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
    kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
    
    
    # gedi
    code_0 = kwargs_gedi['code_0']
    code_1 = kwargs_gedi['code_1']
    nt_id = tokenizer.encode(code_0)[0]
    pt_id = tokenizer.encode(code_1)[0]
    disc_weight = kwargs_gedi['disc_weight']
    prefix_sequence = kwargs_gedi['prefix']
    # prompt
    prefix_sequence = prefix_sequence / prefix_sequence.norm(2, -1, keepdim=True)
    prefix_projections_gedi = gedi_model.clip_project(prefix_sequence).view(-1, gedi_model.prefix_length, gedi_model.gpt_embedding_size)
    prefix_projections_gedi = torch.cat((prefix_projections_gedi, prefix_projections_gedi),dim=0)
    # style
    style = kwargs_gedi['style']
    style_token = torch.tensor([tokenizer.encode(style_pn)[0] for style_pn in style]).reshape(-1, 1).to(device=next(model.parameters()).device,dtype=torch.int64)
    weight = (style_token == pt_id).type_as(style_token).view(-1,1).to(next(model.parameters()).device)
    seq_a = pt_id * weight + nt_id * (1-weight)
    seq_b = nt_id * weight + pt_id * (1-weight)
    seq_batched = torch.cat((seq_a,seq_b),dim=0) 
    embedding_text_style = gedi_model.gpt.transformer.wte(seq_batched) 
    seq_batched_embedding = torch.cat((embedding_text_style, prefix_projections_gedi),dim=1)
    gedi_pad_lens = None
    gedi_past = None
    input_ids = torch.full((batch_size, 1), 50256, dtype=torch.long, device=next(model.parameters()).device)
    vocab_transform = gedi_kwargs['vocab_transform']
    
    # inference word by word
    for t in range(cfg.MODEL.SEQ_LEN):
        # base
        kwargs[cfg.PARAM.WT] = wt
        kwargs[cfg.PARAM.STATE] = state
        logprobs_t, state = model.get_logprobs_state(**kwargs)

        # gedi
        if not gedi_past is None:
            model_inputs = gedi_model.gpt.prepare_inputs_for_generation(input_ids, past=gedi_past)
            input_batched = torch.cat((model_inputs["input_ids"],model_inputs["input_ids"]),dim=0)
            seq_batched = torch.cat((seq_batched,input_batched),dim=1)
            inputs = gedi_model.gpt.prepare_inputs_for_generation(seq_batched, past=gedi_past)
            inputs["pad_lens"] = gedi_pad_lens
        else:
            inputs = {"inputs_embeds": seq_batched_embedding, "pad_lens": gedi_pad_lens, "past":gedi_past}
        gedi_outputs = gedi_model.gpt(**inputs)
        # 剪枝
        gedi_outputs_0_temp_softmax = torch.softmax(gedi_outputs[0], -1)
        threshold = 1e-3
        gedi_outputs_0_temp_softmax[gedi_outputs_0_temp_softmax < threshold] = threshold
        gedi_outputs_0_temp_softmax[gedi_outputs_0_temp_softmax > 0.8] = 0.8
        gedi_outputs = (torch.log(gedi_outputs_0_temp_softmax), gedi_outputs[1])
        # gedi准备
        if gedi_past is None: 
            if gedi_outputs[0].shape[1]>seq_batched_embedding.shape[1]:# 这个没用
                shift_logits = gedi_outputs[0][..., :-1, :].contiguous()
                shift_labels = seq_batched[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                logits_r  = -1*loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                logits_r = logits_r.view(seq_batched.shape[0], -1)
                seq_len = logits_r.shape[1]
                logits_r = torch.sum(logits_r,1)
                logits_pos,logits_neg = torch.split(logits_r/seq_len,input_ids.shape[0])
                logits0 = torch.stack((logits_pos,logits_neg),1)
                if "logit_scale" in dir(gedi_model):
                    logits0 = gedi_model.logit_scale*logits0
                if "bias" in dir(gedi_model):
                    logits0 = logits0 + gedi_model.bias
                # if not (class_bias==0):
                #     logits0[:,0] += class_bias
                logp_desired = torch.log_softmax(logits0,-1)[:,0]
                logp_undesired = torch.log_softmax(logits0,-1)[:,1]
            else:
                seq_len=0
                logp_desired = (torch.zeros(input_ids.shape[0]) + torch.log(torch.tensor(0.5))).to(input_ids.device)
                logp_undesired = (torch.zeros(input_ids.shape[0]) + torch.log(torch.tensor(0.5))).to(input_ids.device)
                logits_r = torch.zeros(input_ids.shape[0]*2).to(input_ids.device)
        # gedi计算
        seq_len= seq_len+1
        gedi_logits= (torch.log_softmax(gedi_outputs[0][:, -1, :],-1)+logits_r.unsqueeze(1))
        logits_pos,logits_neg = torch.split(gedi_logits/seq_len,input_ids.shape[0])
        logits = torch.stack((logits_pos,logits_neg),2)
        if "logit_scale" in dir(gedi_model.gpt):
            logits = gedi_model.gpt.logit_scale*logits
        if "bias" in dir(gedi_model.gpt):
            logits = logits + gedi_model.gpt.bias
        logp_desired_t = torch.log_softmax(logits,-1)[:,:,0]
        logp_undesired_t = torch.log_softmax(logits,-1)[:,:,1]

        # guide
        if t <= 0:
            logprobs_t = torch.log_softmax(1*logprobs_t,-1)
        else:
            logprobs_t = torch.log_softmax(1*logprobs_t,-1) + disc_weight*(vocab_transform.logP_gpt2puret(logp_desired_t))


        # 采样
        if greedy_decode:
            logP_t, wt = torch.max(logprobs_t, 1)
        else:
            probs_t = torch.exp(logprobs_t)
            wt = torch.multinomial(probs_t, 1)
            logP_t = logprobs_t.gather(1, wt)

        # 为下一步准备
        gedi_past = gedi_outputs[1]

        token_gpt = vocab_transform.token_puret2gpt(wt, t==0).to(wt.device)
        token_list = token_gpt.tolist()+token_gpt.tolist()

        for i in range(0,len(token_list)):
            logits_r[i] = gedi_logits[i,token_list[i]]
        for i in range(0,len(token_gpt)):
            logp_desired[i] = logp_desired_t[i,token_gpt[i]]
            logp_undesired[i] = logp_undesired_t[i,token_gpt[i]]
        input_ids = torch.cat([input_ids, token_gpt.unsqueeze(-1)], dim=-1)
        

        wt = wt.view(-1).long()
        unfinished = unfinished * (wt > 0)
        wt = wt * unfinished.type_as(wt)
        sents[:,t] = wt
        logprobs[:,t] = logP_t.view(-1)


        if unfinished.sum() == 0:
            break

    model.decoder.clear_buffer()
    return sents, logprobs
     


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # base model
    parser.add_argument("--pretrained_path", default="/home/liwc/wxp/Alignment/github/trained_model/factual_model/model_pureT_SCST_30.pth")
    # gedi model
    parser.add_argument("--gedi_model_name_or_path", default="/home/liwc/wxp/Alignment/github/trained_model/stylized_model/model_fu_1.pt", type=str)
    parser.add_argument("--code_1", default=" humorous")
    parser.add_argument("--code_0", default=" factual")
    parser.add_argument("--disc_weight", type=float, default=175)
    
    # 数据集参数
    parser.add_argument("--vocab_path", default="./mscoco/txt/coco_vocabulary.txt")
    parser.add_argument('--data_test', default='/home/liwc/wxp/Alignment/github/dataset/FlickrStyle10k/FlickrStyle10k_ViT-L_14_test.pkl')
    parser.add_argument('--teststyle', default='humorous')
    parser.add_argument('--max_seq_len', default=17)
    parser.add_argument("--batch_size", default=36)
    # 保存
    parser.add_argument("--generated_path", default="./generated/guide")
    parser.add_argument("--gen_model_type", default="pureT_SCST")
    # 设置
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    print("generate parameters %s", args)

    # base model
    cfg_from_file("./experiments_PureT/PureT_SCST/config.yml")
    # cfg.ROOT_DIR = "./experiments_PureT/PureT_SCST/"
    model = PureT()
    model = model.to(args.device)
    model.load_state_dict(torch.load(args.pretrained_path ,map_location=lambda storage, loc: storage))

    # gedi model
    args.class_bias = 0.0
    config_class, model_class, tokenizer_class = GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained("gpt2", do_lower_case=False)
    config = config_class.from_pretrained("gpt2")
    config.nbias = 0
    config.logit_scale = True
    gpt_gedi = model_class.from_pretrained("gpt2", config=config)
    gedi_model = ClipCaptionModel(tokenizer, gpt_gedi, 4, prefix_size=768)
    gedi_model.load_state_dict(torch.load(args.gedi_model_name_or_path, map_location="cpu"))
    gedi_model.to(args.device)


    # dataset
    dataset = RCStyleDataset_infer(vocab_path = args.vocab_path, data_path = args.data_test, max_seq_len = args.max_seq_len, teststyle=args.teststyle)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    vocab_transform = vocab_gpt2puret(tokenizer.decoder, tokenizer.encoder, dataset.vocab)

    # generate
    model.eval()

    # weight_set = [0, 25, 50, 75, 100, 125, 150, 175, 200]
    # weight_set = [0, 48, 10, 20, 30, 40, 50, 60, 70, 80]
    # for weight in weight_set:
        # args.disc_weight = weight
    if True:
        gens = []
        refs = []
        image_paths = []

        with torch.no_grad():
            for infer_step, (imgid, filename, image_embedding, att_feats, style) in enumerate(tqdm.tqdm(loader, desc="test")):

                att_feats = att_feats.to(next(model.parameters()).device)
                image_embedding = image_embedding.to(next(model.parameters()).device, dtype=torch.float32)
    
                # make_kwargs
                base_kwargs = {}
                base_kwargs['GV_FEAT'] = None
                base_kwargs['ATT_FEATS'] = att_feats
                base_kwargs['ATT_FEATS_MASK'] = None
                base_kwargs['BEAM_SIZE'] = 1
                base_kwargs['GREEDY_DECODE'] = True

                gedi_kwargs = {}
                gedi_kwargs['code_0'] = args.code_0
                gedi_kwargs['code_1'] = args.code_1
                gedi_kwargs['style'] = style
                gedi_kwargs['disc_weight'] = args.disc_weight
                gedi_kwargs['prefix'] = image_embedding
                gedi_kwargs['vocab_transform'] = vocab_transform

                seq, _ = decode_generate(model, base_kwargs, gedi_model, gedi_kwargs)
                # if kwargs['BEAM_SIZE'] > 1:
                #     seq, _ = model.decode_beam(**kwargs)
                # else:
                #     seq, _ = model.decode(**kwargs)


                
                sents = utils.decode_sequence(dataset.vocab, seq.data)
                imgid = imgid.numpy()
                for sid, sent in enumerate(sents):
                    gens.append(sent)
                    refs.append(dataset.caption[int(imgid[sid])])
                image_paths.extend(filename)


        # eval
        result = {}
        clipmodel, preprocess = clip.load('ViT-L/14', device=args.device, jit=False)
        clipscores, other_metrics = computer_clipscore_and_other(image_paths, clipmodel, args.device, gens, refs)
        result.update(clipscores)
        result.update(other_metrics)

        # 保存描述
        generate_path = args.generated_path + "/" + args.gen_model_type + "/" + args.teststyle
        if not os.path.exists(generate_path):
            os.makedirs(generate_path)
            
        out_txt_dir = args.generated_path + "/" + args.gen_model_type + "/" + args.teststyle +"/captions_generate_"+ str(args.disc_weight) + ".txt"
        with open(out_txt_dir, "w") as file:
            for generate_ref in gens:
                file.write(generate_ref + "\n")

        # 计算ppl评估指标
        ppl_out_path = args.generated_path + "/" + args.gen_model_type + "/" + args.teststyle + "/ppl_out_" + str(args.disc_weight) + ".txt"
        result["ppl"] = eval_ppl(out_txt_dir, args.teststyle, ppl_out_path)

        # 计算acc评估指标
        file_error = args.generated_path + "/" + args.gen_model_type + "/" + args.teststyle + "/file_error_"+str(args.disc_weight)+".txt"
        result["acc"] = eval_acc(out_txt_dir, args.teststyle, args.device, tokenizer, file_error)

        
        for key, value in result.items():
                if isinstance(value, np.float16):
                    result[key] = float(value)
        print(json.dumps({**result}))
        print("Its weight is :" + str(args.disc_weight))