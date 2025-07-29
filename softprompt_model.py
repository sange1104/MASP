import torch
import torch.nn as nn

VIEW_NAMES = ["facial_expression", "object", "scene", "human_action", "brightness", "colorfulness"]
NUM_SOFT_PROMPT_TOKENS = 250

def load_pretrained_model(model_name: str = "qwen"):
    """
    Load the model and its corresponding processor.

    Args:
        model_name (str): Hugging Face model identifier.

    Returns:
        model: The loaded vision-to-sequence model.
        processor: The corresponding processor for image + text inputs.
    """
    if model_name == "llava": 
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        model = LlavaNextForConditionalGeneration.from_pretrained(model_name) 
        processor = LlavaNextProcessor.from_pretrained(model_name)
        model.config.pad_token_id = processor.tokenizer.eos_token_id 
        
    elif model_name == "qwen":
        from transformers import AutoModelForVision2Seq, AutoProcessor
        model_name = "Qwen/Qwen2-VL-7B-Instruct"
        model = AutoModelForVision2Seq.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name) 
    
    elif model_name == "blip3":
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        model_name = "Salesforce/instructblip-vicuna-7b" 
        model = InstructBlipForConditionalGeneration.from_pretrained(model_name)
        processor = InstructBlipProcessor.from_pretrained(model_name)
     
    return model, processor

class AttentionFuser(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),    
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)            
        )

    def forward(self, z_all):
        """
        z_all: (B, 6, D)
        return: (B, 1, D)
        """
        scores = self.score_net(z_all).squeeze(-1)         
        attn_weights = torch.softmax(scores, dim=1)         
        fused = torch.sum(z_all * attn_weights.unsqueeze(-1), dim=1, keepdim=True)   
        return fused
    
class SoftPromptEmotionModel(nn.Module):
    def __init__(self, base_model, processor, stage="stage1"):
        super().__init__()
        self.vlm = base_model
        self.processor = processor
        self.tokenizer = processor.tokenizer 
        self.embedding = self.vlm.get_input_embeddings()
        self.stage = stage
        self.image_token_dropout_prob = 0.

        for p in self.vlm.parameters():
            p.requires_grad = False
        N = 5
        self.q_view = nn.ParameterDict({
            view: nn.Parameter(torch.randn(N, self.embedding.embedding_dim))
            for view in VIEW_NAMES
        })

        self.cross_attn = nn.ModuleDict({
            view: nn.MultiheadAttention(embed_dim=self.embedding.embedding_dim, num_heads=1, batch_first=True)
            for view in VIEW_NAMES
        })

        self.fuser = AttentionFuser(self.embedding.embedding_dim)

        for name, param in self.named_parameters():
            if stage == "stage1":
                param.requires_grad = "q_view" in name or "cross_attn" in name
            elif stage == "stage2":  
                param.requires_grad = False
                
        for p in self.q_view.parameters():
            if stage == "stage2":  
                p.requires_grad = False

    def _replace_vision_tokens_stage1(self, ids, mask, z, input_device):
        """
        Replace vision token region with view-specific embedding z for stage1.
        Returns: input_embeds, attn_mask, label
        """
        # Find <vision_start> and <vision_end>
        vision_start_idx = (ids == 151652).nonzero(as_tuple=True)[0][0].item()
        vision_end_idx = (ids == 151653).nonzero(as_tuple=True)[0][-1].item()

        # Split input_ids
        ids_before = ids[:vision_start_idx]
        ids_after = ids[vision_end_idx+1:]

        # Embedding
        embed_before = self.vlm.get_input_embeddings()(ids_before)
        embed_after = self.vlm.get_input_embeddings()(ids_after)
        embed_vs = self.vlm.get_input_embeddings()(ids[vision_start_idx].unsqueeze(0))  # (1, D)
        embed_ve = self.vlm.get_input_embeddings()(ids[vision_end_idx].unsqueeze(0))    # (1, D)
        embed = torch.cat([embed_before, embed_vs, z, embed_ve, embed_after], dim=0)

        # Attention mask
        mask_before = mask[:vision_start_idx]
        mask_after = mask[vision_end_idx+1:]
        mask_vs = mask[vision_start_idx].unsqueeze(0)
        mask_ve = mask[vision_end_idx].unsqueeze(0)
        attn = torch.cat([
            mask_before,
            mask_vs,
            torch.ones((z.size(0),), dtype=torch.long, device=input_device),
            mask_ve,
            mask_after
        ])

        # Labels
        label = ids.clone()
        label[:] = -100
        z_len = z.size(0)
        label_vs = torch.full((1,), -100, dtype=torch.long, device=self.vlm.device)
        label_ve = torch.full((1,), -100, dtype=torch.long, device=self.vlm.device)
        label_z = torch.full((z_len,), -100, dtype=torch.long, device=self.vlm.device)
        label = torch.cat([
            label[:vision_start_idx],
            label_vs,
            label_z,
            label_ve,
            label[vision_end_idx + 1:]
        ], dim=0)

        # Restore answer tokens
        start_idx_orig = (ids == 77091).nonzero(as_tuple=True)[0].item() + 1
        end_idx_orig = (ids == 151645).nonzero(as_tuple=True)[0][-1].item()
        num_vision_tokens = vision_end_idx - vision_start_idx + 1
        new_token_len = z_len + 2
        shift = num_vision_tokens - new_token_len

        def map_index(old_idx):
            if old_idx < vision_start_idx:
                return old_idx
            elif old_idx > vision_end_idx:
                return old_idx - shift
            else:
                return None

        start_idx = map_index(start_idx_orig)
        end_idx = map_index(end_idx_orig)
        if start_idx is not None and end_idx is not None:
            label[start_idx:end_idx] = ids[start_idx_orig:end_idx_orig]

        return embed, attn, label

    def _replace_vision_tokens_stage2(self, ids, mask, z, vision_feats, input_device):
        """
        Replace vision token region with fused z and vision_feats for stage2.
        Returns: input_embeds, attn_mask, label
        """
        vision_start_idx = (ids == 151652).nonzero(as_tuple=True)[0][0].item()
        vision_end_idx = (ids == 151653).nonzero(as_tuple=True)[0][-1].item()

        ids_before = ids[:vision_start_idx]
        ids_after = ids[vision_end_idx+1:]

        embed_before = self.vlm.get_input_embeddings()(ids_before)
        embed_after = self.vlm.get_input_embeddings()(ids_after)
        embed_vs = self.vlm.get_input_embeddings()(ids[vision_start_idx].unsqueeze(0))
        embed_ve = self.vlm.get_input_embeddings()(ids[vision_end_idx].unsqueeze(0))
        embed = torch.cat([
            embed_before,
            embed_vs,
            z,
            vision_feats,
            embed_ve,
            embed_after
        ], dim=0)

        mask_before = mask[:vision_start_idx]
        mask_after = mask[vision_end_idx+1:]
        mask_vs = mask[vision_start_idx].unsqueeze(0)
        mask_ve = mask[vision_end_idx].unsqueeze(0)
        prompt_mask = torch.ones((z.size(0)+vision_feats.size(0),), dtype=torch.long, device=input_device)
        attn = torch.cat([mask_before, mask_vs, prompt_mask, mask_ve, mask_after])

        label = ids.clone()
        label[:] = -100
        label_vs = torch.full((1,), -100, dtype=torch.long, device=input_device)
        label_ve = torch.full((1,), -100, dtype=torch.long, device=input_device)
        label_z = torch.full((z.size(0)+vision_feats.size(0),), -100, dtype=torch.long, device=input_device)
        label = torch.cat([
            label[:vision_start_idx],
            label_vs,
            label_z,
            label_ve,
            label[vision_end_idx+1:]
        ], dim=0)

        start_idx_orig = (ids == 77091).nonzero(as_tuple=True)[0].item() + 1
        end_idx_orig = (ids == 151645).nonzero(as_tuple=True)[0][-1].item()
        old_vision_len = vision_end_idx - vision_start_idx + 1
        new_vision_len = z.size(0) + vision_feats.size(0) + 2
        shift = old_vision_len - new_vision_len

        def map_index(old_idx):
            if old_idx < vision_start_idx:
                return old_idx
            elif old_idx > vision_end_idx:
                return old_idx - shift
            else:
                return None

        start_idx = map_index(start_idx_orig)
        end_idx = map_index(end_idx_orig)
        if start_idx is not None and end_idx is not None:
            label[start_idx:end_idx] = ids[start_idx_orig:end_idx_orig]

        return embed, attn, label

    def forward(self, **kwargs):
        """
        Forward pass for both stage1 (single view) and stage2 (multi-view fusion).
        Returns: loss, outputs
        """
        pixel_values = kwargs["pixel_values"]
        messages = kwargs["messages"]
        image_grid_thw = kwargs["image_grid_thw"]
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        input_device = pixel_values.device
        B = len(messages)

        if self.stage == 'stage1':
            task = kwargs["task"]
            vision_feats = self.vlm.visual(pixel_values, grid_thw=image_grid_thw)
            z_list = []
            for i in range(B):
                view_name = task[i]
                q = self.q_view[view_name]
                z_i, _ = self.cross_attn[view_name](q, vision_feats, vision_feats)
                z_list.append(z_i)
            z_stack = torch.stack(z_list, dim=0)

            input_embeds, attn_mask, labels_list = [], [], []
            for i in range(B):
                embed, attn, label = self._replace_vision_tokens_stage1(
                    input_ids[i], attention_mask[i], z_stack[i], input_device
                )
                input_embeds.append(embed)
                attn_mask.append(attn)
                labels_list.append(label)

            input_embeds = torch.nn.utils.rnn.pad_sequence(input_embeds, batch_first=True)
            attn_mask = torch.nn.utils.rnn.pad_sequence(attn_mask, batch_first=True)
            labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True)

            outputs = self.vlm(
                inputs_embeds=input_embeds,
                attention_mask=attn_mask,
                labels=labels,
                return_dict=True
            )

        elif self.stage == 'stage2':
            vision_feats = self.vlm.visual(pixel_values, grid_thw=image_grid_thw)
            z_all = []
            for i in range(B):
                z_i_list = []
                for view in VIEW_NAMES:
                    q = self.q_view[view]
                    z, _ = self.cross_attn[view](q, vision_feats, vision_feats)
                    z_i_list.append(z.squeeze(0))
                z_i = torch.cat(z_i_list, dim=0)
                z_all.append(z_i)
            z_all = torch.stack(z_all, dim=0)

            input_embeds, attn_mask, labels_list = [], [], []
            for i in range(B):
                embed, attn, label = self._replace_vision_tokens_stage2(
                    input_ids[i], attention_mask[i], z_all[i], vision_feats, input_device
                )
                input_embeds.append(embed)
                attn_mask.append(attn)
                labels_list.append(label)

            input_embeds = torch.nn.utils.rnn.pad_sequence(input_embeds, batch_first=True)
            attn_mask = torch.nn.utils.rnn.pad_sequence(attn_mask, batch_first=True)
            labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True)

            outputs = self.vlm(
                inputs_embeds=input_embeds,
                attention_mask=attn_mask,
                labels=labels,
                return_dict=True
            )

        return outputs.loss, outputs

    @torch.no_grad()
    def generate(self, **kwargs):
        """
        Generate method for both stage1 (single view) and stage2 (multi-view fusion).
        Returns: generated sequences
        """
        pixel_values = kwargs["pixel_values"]
        image_grid_thw = kwargs["image_grid_thw"]
        input_ids = kwargs["input_ids"]
        task = kwargs["task"]
        attention_mask = kwargs["attention_mask"]
        input_device = pixel_values.device
        vision_feats = self.vlm.visual(pixel_values, grid_thw=image_grid_thw)
        vision_feats = vision_feats.unsqueeze(0)
        B = len(task)

        if self.stage == 'stage1':
            full_embeds, full_attn_masks = [], []
            for i in range(B):
                view_name = task[i]
                q = self.q_view[view_name]
                z, _ = self.cross_attn[view_name](
                    q.unsqueeze(0), vision_feats, vision_feats
                )
                z_embed = z.squeeze(0)
                ids = input_ids[i]
                mask = attention_mask[i]

                embed, attn, _ = self._replace_vision_tokens_stage1(
                    ids, mask, z_embed, input_device
                )
                full_embeds.append(embed.unsqueeze(0))
                full_attn_masks.append(attn.unsqueeze(0))

            full_embeds = torch.cat(full_embeds, dim=0)
            full_attn_masks = torch.cat(full_attn_masks, dim=0)

            generate_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in {"pixel_values", "messages", "image_grid_thw", "input_ids", "task", "attention_mask"}
            }

            return self.vlm.generate(
                inputs_embeds=full_embeds,
                attention_mask=full_attn_masks,
                max_new_tokens=64,
                temperature=0.8,
                do_sample=True,
                top_p=1.0,
                **generate_kwargs
            )

        elif self.stage == 'stage2':
            z_i_list = []
            for view in VIEW_NAMES:
                q = self.q_view[view]
                z, _ = self.cross_attn[view](q.unsqueeze(0), vision_feats, vision_feats)
                z_i_list.append(z.squeeze(0))
            z_i = torch.cat(z_i_list, dim=0)
            vision_feats_ = vision_feats[0]
            ids = input_ids[0]
            mask = attention_mask[0]

            embed, attn, _ = self._replace_vision_tokens_stage2(
                ids, mask, z_i, vision_feats_, input_device
            )
            full_embeds = embed.unsqueeze(0)
            full_attn_mask = attn.unsqueeze(0)

            generate_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in {"pixel_values", "messages", "image_grid_thw", "input_ids", "task", "attention_mask"}
            }

            return self.vlm.generate(
                inputs_embeds=full_embeds,
                attention_mask=full_attn_mask,
                temperature=0.8,
                do_sample=False,
                **generate_kwargs
            )

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.vlm, "gradient_checkpointing_enable"):
            self.vlm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.vlm, "gradient_checkpointing_disable"):
            self.vlm.gradient_checkpointing_disable()
             

def prepare_model_and_tokenizer():
    """Load model and tokenizer, add soft prompt tokens, and resize embeddings."""
    model, processor = load_pretrained_model()
    prefix_token_strs = [f"<|reserved_special_token_{i}|>" for i in range(NUM_SOFT_PROMPT_TOKENS)]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": prefix_token_strs})
    model.resize_token_embeddings(len(processor.tokenizer))
    prefix_token_ids = processor.tokenizer.convert_tokens_to_ids(prefix_token_strs)
    return model, processor, prefix_token_ids, prefix_token_strs


def register_embedding_gradient_mask(model, prefix_token_ids, num_special_tokens_in_prefix):
    """Register gradient hook to update only soft prompt embeddings."""
    model.get_input_embeddings().weight.requires_grad = True
    embeddings_to_update = torch.tensor(prefix_token_ids[:num_special_tokens_in_prefix], dtype=torch.long).to(model.device)
    def grad_hook(grad):
        mask = torch.zeros_like(grad)
        mask[embeddings_to_update] = 1.0
        return grad * mask
    hook_handle = model.get_input_embeddings().weight.register_hook(grad_hook)
    return hook_handle

def initialize_and_load_softprompt_model(model, processor, ckpt_path):
    """Initialize SoftPromptEmotionModel and load parameters from checkpoint."""
    spmodel = SoftPromptEmotionModel(model, processor, stage="stage2").to(torch.device('cuda'))
    embedding = spmodel.vlm.get_input_embeddings()
    embedding.weight.requires_grad = True

    ckpt = torch.load(ckpt_path)
    for k in spmodel.q_view:
        spmodel.q_view[k].data.copy_(ckpt["q_view"][k].to(spmodel.q_view[k].device))
    for k in spmodel.cross_attn:
        spmodel.cross_attn[k].load_state_dict(ckpt["cross_attn"][k])
    return spmodel

def save_soft_prompt_embeddings(model, processor, prefix_token_ids, num_special_tokens_in_prefix, save_ckpt):
    """Save soft prompt embeddings to checkpoint file."""
    embedding_weights = model.get_input_embeddings().weight.data
    saved_embeddings = {}
    token_ids = prefix_token_ids[:num_special_tokens_in_prefix]
    token_strs = processor.tokenizer.convert_ids_to_tokens(token_ids)
    for token_str, token_id in zip(token_strs, token_ids):
        saved_embeddings[token_str] = embedding_weights[token_id].cpu()
    if save_ckpt is not None:
        torch.save(saved_embeddings, save_ckpt)
 
def load_soft_prompt_embeddings(ckpt_path, tokenizer, model):
    """
    Load soft prompt embeddings from a checkpoint and update the model's embedding weights.
    Returns the list of prefix token strings.
    """
    loaded_embeddings = torch.load(ckpt_path, map_location='cuda')
    prefix_token_strs = list(loaded_embeddings.keys())
    tokenizer.add_special_tokens({"additional_special_tokens": prefix_token_strs})
    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        for token_str, embedding in loaded_embeddings.items():
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id == tokenizer.unk_token_id:
                raise ValueError(f"{token_str} was not properly registered.")
            model.get_input_embeddings().weight[token_id] = embedding.to(model.device)

    return prefix_token_strs