from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import re
import os.path as osp

import torch
from torch import nn
from modules.tokenization_clip import SimpleTokenizer
from mmengine.runner.amp import autocast

from modules.until_module import AllGather, CrossEn
from modules.module_clip import CLIP, convert_weights
from registry import MODELS

logger = logging.getLogger(__name__)
allgather = AllGather.apply


SPECIAL_TOKENS = {
    "CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
    "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"
}

_PT_NAME = {
    "ViT-B/32": "ViT-B-32.pt",
    "v32": "ViT-B-32.pt",
    "ViT-B/16": "ViT-B-16.pt",
    "v16": "ViT-B-16.pt",
    "ViT-L/14": "ViT-L-14.pt",
    "vl14": "ViT-L-14.pt",
}


@MODELS.register_module()
class BaselineModel(nn.Module):
    # def init(self, *params, **ka):
    def __init__(
        self, 
        clip_arch='v16',
        adapter={},
        checkpoint=None,
        max_words=32,
        **kwargs
    ):
        super().__init__()
        self.tokenizer = SimpleTokenizer('modules/bpe_simple_vocab_16e6.txt.gz')
        self.cache_dir = kwargs.get('clip_cache_dir', '.cache/clip')

        self.epoch = 0
        self.checkpoint = checkpoint
        self.max_words = max_words
        self.load_clip(clip_arch, adapter)
        self.fp32 = kwargs.get('fp32', False)
        self.training_type = torch.float32 if self.fp32 else torch.float16
        
        self.loss_fct = CrossEn()
        
        # super().__init__(*params)

    def load_clip(self, clip_arch, adapter):
        state_dict = {}
        model_path = osp.join(self.cache_dir, _PT_NAME[clip_arch])
        clip_model = torch.jit.load(model_path, map_location="cpu").eval()
        clip_state_dict = clip_model.state_dict()

        attn_pattern = r'resblocks\.[0-9]+\.attn.in_proj'
        ln1_pattern = r'resblocks\.[0-9]+\.ln_1'
        ln2_pattern = r'resblocks\.[0-9]+\.ln_2'
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if re.search(attn_pattern, new_key):
                qkv = val.reshape(3, -1, *val.shape[1:]).clone()
                for i, name in enumerate(['q', 'k', 'v']):
                    state_dict[new_key.replace('attn.in_proj_', f'attn.{name}_proj.')] = qkv[i]
                continue
            elif re.findall(ln1_pattern, new_key):
                prefix = re.findall(ln1_pattern, new_key)[0]
                new_key = new_key.replace(prefix, f'{prefix[:-5]}.attn.ln')
            elif re.findall(ln2_pattern, new_key):
                prefix = re.findall(ln2_pattern, new_key)[0]
                new_key = new_key.replace(prefix, f'{prefix[:-5]}.mlp.ln')

            state_dict[new_key] = val.clone()

        vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
            adapter
        ).float()
    
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self, prefix='')

        convert_weights(self.clip)

    def to_device(self, tensor):
        if isinstance(tensor, list):
            tensor = torch.stack([self.to_device(t) for t in tensor])
        return tensor.to(self.clip.logit_scale.device)

    def process_caption(self, caption_batch):
        """
        :param caption_batch: list[str]
        :return torch.Tensor (bs, max_words, embed_dim)
        """
        max_words = self.max_words
        tokens = [
            [SPECIAL_TOKENS['CLS_TOKEN']] + 
            self.tokenizer.tokenize(c)[:max_words - 2] + 
            [SPECIAL_TOKENS["SEP_TOKEN"]]
        for c in caption_batch]
        token_ids = [
            self.tokenizer.convert_tokens_to_ids(t) +
            [0] * (max_words - len(t))
        for t in tokens]
        caption_tensors = self.to_device(torch.tensor(token_ids))
        caption_masks = torch.ones_like(caption_tensors)
        caption_masks[caption_tensors == 0] = 0
        return caption_tensors, caption_masks

    def forward(self, batch):
        with autocast(dtype=self.training_type if self.training else torch.float32):
            videos, captions, v_mask = batch['video_batch'], batch['caption_batch'], batch['v_mask']
            texts, t_mask = self.process_caption(captions)
            videos = self.to_device(torch.stack(videos))
            v_mask = self.to_device(v_mask)
            videos = videos.reshape(-1, *videos.shape[2:])
            if self.training:
                loss = 0.
                sequence_output, visual_output = self.get_sequence_visual_output(texts, t_mask, videos, v_mask)

                t2v_logits, v2t_logits = self.get_similarity_logits(
                    sequence_output, visual_output, t_mask, v_mask
                )
                sim_loss1 = self.loss_fct(t2v_logits)
                sim_loss2 = self.loss_fct(v2t_logits)
                sim_loss = (sim_loss1 + sim_loss2) / 2
                loss += sim_loss
                return loss

            else: 
                sequence_output, visual_output = self.get_sequence_visual_output(texts, t_mask, videos, v_mask)
                return sequence_output, visual_output, t_mask, v_mask
    
    def get_similarity_logits(self, t_emb, v_emb, t_mask, v_mask):
        if isinstance(t_emb, list):
            t_emb, v_emb = torch.cat(t_emb), torch.cat(v_emb)
        t_emb, v_emb = t_emb.contiguous(), v_emb.contiguous()
        if self.training:
            v_emb = allgather(v_emb)
            v_mask = allgather(v_mask)
            t_emb = allgather(t_emb)
            torch.distributed.barrier()

        t_emb = t_emb.squeeze(1)
        t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
        logit_scale = self.clip.logit_scale.exp().to(v_emb.device)
        
        v_emb = v_emb / v_emb.norm(dim=-1, keepdim=True)
        v_emb = self._mean_pooling_for_similarity_visual(v_emb, v_mask)
        v_emb = v_emb / v_emb.norm(dim=-1, keepdim=True)
        logits = logit_scale * torch.matmul(t_emb, v_emb.t())
        return logits, logits.t()

    def get_visual_output(self, video, video_mask):
 
        visual_output = self.clip.encode_image(video, mask=video_mask).float()
        visual_output = visual_output.view(*video_mask.shape, -1)
        return visual_output
    
    def get_sequence_output(self, text, t_mask):
        bs_pair = text.size(0)
        sequence_hidden = self.clip.encode_text(text, mask=t_mask).float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden

    def get_sequence_visual_output(self, text, t_mask, video, v_mask):

        sequence_output = self.get_sequence_output(text, t_mask)
        visual_output = self.get_visual_output(video, v_mask)
        return sequence_output, visual_output

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out
