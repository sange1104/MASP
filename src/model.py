import torch  
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForVision2Seq 

def load_vlm_model(model_name, device_map="cuda"):
    '''
    Load a Vision-Language Model (VLM) and its processor, and freeze all parameters.

    This utility function:
        - Loads a pretrained Qwen2-VL (or compatible) Vision2Seq model. 
        - Freezes all model parameters to prevent any gradient update. 

    Args:
        model_name (str):
            Name or path of the Hugging Face VLM checkpoint.
            Example: 'Qwen/Qwen2-VL-7B-Instruct'
        
        device_map (str or dict):
            Device placement strategy. Defaults to "cuda".

    Returns:
        model (PreTrainedModel):
            Loaded Vision2Seq model with all parameters set to requires_grad=False.
        processor (AutoProcessor):
            Corresponding processor that handles multimodal tokenization, image transforms,
            and prepares inputs for the model.
    '''

    # Load the pretrained Vision-Language model and place its weights across devices.
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        device_map=device_map
    )

    # Freeze all model parameters  
    for param in model.parameters():
        param.requires_grad = False

    # Load the corresponding multimodal processor for text/image encoding
    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor

class MASPEmotionModel(nn.Module):
    '''
    Wrapper model that augments a frozen Vision-Language Model (VLM) with
    view-specific soft prompts and cross-attention over visual features.

    The model has two stages:
        - Stage 1: Learn view-specific query prompts that attend to visual features.
          Each sample uses a single target view (scene, object, etc.).
        - Stage 2: Fuse multiple view prompts (across all view types) and replace
          the vision token region with a concatenation of all view prompts
          and visual features, then train a high-level fusion behavior.

    The base VLM remains frozen in both stages. Only view queries and cross-attn
    modules are trainable in Stage 1; in Stage 2, the model is typically used
    in a frozen/inference-like manner (depending on your training setup).
    '''
    def __init__(self, base_model, processor, view_names, stage="stage1"):
        super().__init__()
        self.vlm = base_model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.embedding = self.vlm.get_input_embeddings()
        self.stage = stage
        self.view_names = view_names
        self.image_token_dropout_prob = 0.0  # Placeholder for potential future use

        # 1) Freeze the entire base VLM by default
        self._freeze_vlm_parameters()

        # 2) Initialize view-specific query prompts and cross-attention modules
        self._init_view_prompts_and_cross_attn()

        # 3) Configure which parameters are trainable depending on stage
        self._configure_trainable_parameters(stage)

    def _freeze_vlm_parameters(self):
        '''
        Freeze all parameters of the underlying VLM so that only the
        added modules (e.g., view prompts, cross-attention) can be trained.
        '''
        for p in self.vlm.parameters():
            p.requires_grad = False

    def _init_view_prompts_and_cross_attn(self, num_query_tokens=5, num_heads=1):
        '''
        Initialize learnable view-specific query prompts and cross-attention modules.

        Args:
            num_query_tokens (int): Number of query tokens per view.
            num_heads (int): Number of attention heads for MultiheadAttention.
        '''
        N = num_query_tokens

        # Learnable query matrix for each view: (N, D)
        self.q_view = nn.ParameterDict({
            view: nn.Parameter(torch.randn(N, self.embedding.embedding_dim))
            for view in self.view_names
        })

        # View-specific cross-attention modules (query: q_view, key/value: vision features)
        self.cross_attn = nn.ModuleDict({
            view: nn.MultiheadAttention(
                embed_dim=self.embedding.embedding_dim,
                num_heads=num_heads,
                batch_first=True
            )
            for view in self.view_names
        })

    def _configure_trainable_parameters(self, stage):
        '''
        Set requires_grad flags according to the training stage.

        Stage 1:
            - Only q_view and cross_attn parameters are trainable.
        Stage 2:
            - By default, all parameters are frozen (you can customize this
              if you want to train an extra fusion module).
        '''
        for name, param in self.named_parameters():
            if stage == "stage1":
                # Train only view queries and cross-attention modules
                param.requires_grad = ("q_view" in name) or ("cross_attn" in name)
            elif stage == "stage2":
                # Freeze everything (can be relaxed if you add extra modules)
                param.requires_grad = False

        # Explicitly ensure q_view is frozen in Stage 2
        for p in self.q_view.parameters():
            if stage == "stage2":
                p.requires_grad = False

    def _find_vision_token_range(self, ids):
        '''
        Locate the start and end indices of the vision token region in input_ids.

        This assumes:
            - <vision_start> token id: 151652
            - <vision_end> token id:   151653

        Args:
            ids (torch.LongTensor): 1D tensor of token ids.

        Returns:
            (int, int): (vision_start_idx, vision_end_idx)
        '''
        vision_start_idx = (ids == 151652).nonzero(as_tuple=True)[0][0].item()
        vision_end_idx = (ids == 151653).nonzero(as_tuple=True)[0][-1].item()
        return vision_start_idx, vision_end_idx

    def _find_answer_token_range(self, ids):
        '''
        Locate the token range corresponding to the answer region in input_ids.

        This assumes:
            - Answer starts after a special "answer start" token id: 77091
            - Answer ends before the last occurrence of an "answer end" token id: 151645

        Args:
            ids (torch.LongTensor): 1D tensor of token ids.

        Returns:
            (int, int): (start_idx_orig, end_idx_orig)
        '''
        start_idx_orig = (ids == 77091).nonzero(as_tuple=True)[0].item() + 1
        end_idx_orig = (ids == 151645).nonzero(as_tuple=True)[0][-1].item()
        return start_idx_orig, end_idx_orig

    def _map_index_after_vision_replacement(self, old_idx, vision_start_idx, vision_end_idx, shift):
        '''
        Map an original token index to the new index after the vision token block
        has been replaced by a different-length segment.

        Args:
            old_idx (int): Original index.
            vision_start_idx (int): Start index of original vision region.
            vision_end_idx (int): End index of original vision region.
            shift (int): Number of tokens removed (positive) or added (negative)
                         when replacing the vision region.

        Returns:
            int or None: New index if it remains a valid non-vision index,
                         otherwise None if the original index was inside the vision region.
        '''
        if old_idx < vision_start_idx:
            return old_idx
        elif old_idx > vision_end_idx:
            return old_idx - shift
        else:
            # Token was inside the vision region and no longer maps to a single index
            return None

    def _replace_vision_tokens_stage1(self, ids, mask, z, input_device):
        '''
        Replace the vision token block with view-specific embeddings for Stage 1.

        For Stage 1, the replacement block is:
            [<vision_start>, z (view prompt tokens), <vision_end>]

        The answer tokens are restored into the label tensor while all other
        tokens (including vision-related tokens) are masked out with -100.

        Args:
            ids (torch.LongTensor): 1D input_ids for a single example.
            mask (torch.LongTensor): 1D attention_mask for the same example.
            z (torch.FloatTensor): 2D tensor of view-specific embeddings (Lz, D).
            input_device (torch.device): Device to allocate any new tensors.

        Returns:
            (input_embeds, attn_mask, labels):
                - input_embeds (torch.FloatTensor): Modified input embeddings.
                - attn_mask (torch.LongTensor): Modified attention mask.
                - labels (torch.LongTensor): Label tensor with answer tokens only.
        '''
        # 1) Locate original vision token span
        vision_start_idx, vision_end_idx = self._find_vision_token_range(ids)

        # 2) Split ids and compute embeddings around the vision region
        ids_before = ids[:vision_start_idx]
        ids_after = ids[vision_end_idx + 1:]

        embed_before = self.vlm.get_input_embeddings()(ids_before)
        embed_after = self.vlm.get_input_embeddings()(ids_after)
        embed_vs = self.vlm.get_input_embeddings()(ids[vision_start_idx].unsqueeze(0))  # <vision_start>
        embed_ve = self.vlm.get_input_embeddings()(ids[vision_end_idx].unsqueeze(0))    # <vision_end>

        # Concatenate: [before, <vs>, z, <ve>, after]
        embed = torch.cat([embed_before, embed_vs, z, embed_ve, embed_after], dim=0)

        # 3) Build new attention mask aligned with new embedding sequence
        mask_before = mask[:vision_start_idx]
        mask_after = mask[vision_end_idx + 1:]
        mask_vs = mask[vision_start_idx].unsqueeze(0)
        mask_ve = mask[vision_end_idx].unsqueeze(0)

        # All z tokens are visible (1s in attention mask)
        z_mask = torch.ones((z.size(0),), dtype=torch.long, device=input_device)
        attn = torch.cat([mask_before, mask_vs, z_mask, mask_ve, mask_after])

        # 4) Build label tensor: mask out everything except the final answer region
        label = ids.clone()
        label[:] = -100  # Start by masking everything

        z_len = z.size(0)
        label_vs = torch.full((1,), -100, dtype=torch.long, device=self.vlm.device)
        label_ve = torch.full((1,), -100, dtype=torch.long, device=self.vlm.device)
        label_z = torch.full((z_len,), -100, dtype=torch.long, device=self.vlm.device)

        # Concatenate labels in the same layout as embeddings
        label = torch.cat([
            label[:vision_start_idx],
            label_vs,
            label_z,
            label_ve,
            label[vision_end_idx + 1:]
        ], dim=0)

        # 5) Restore answer tokens in the new label positions
        start_idx_orig, end_idx_orig = self._find_answer_token_range(ids)

        num_vision_tokens = vision_end_idx - vision_start_idx + 1
        new_token_len = z_len + 2  # <vs> + z tokens + <ve>
        shift = num_vision_tokens - new_token_len

        # Map original answer index range onto the new sequence
        start_idx = self._map_index_after_vision_replacement(
            start_idx_orig, vision_start_idx, vision_end_idx, shift
        )
        end_idx = self._map_index_after_vision_replacement(
            end_idx_orig, vision_start_idx, vision_end_idx, shift
        )

        if start_idx is not None and end_idx is not None:
            label[start_idx:end_idx] = ids[start_idx_orig:end_idx_orig]

        return embed, attn, label

    def _replace_vision_tokens_stage2(self, ids, mask, z, vision_feats, input_device):
        '''
        Replace the vision token block with fused view prompts and visual features for Stage 2.

        For Stage 2, the replacement block is:
            [<vision_start>, z (all-view prompts) + vision_feats, <vision_end>]

        Both z and vision_feats are treated as prompt-like tokens that replace
        the original vision token region. Answer tokens are restored in the label
        tensor similarly to Stage 1.

        Args:
            ids (torch.LongTensor): 1D input_ids for a single example.
            mask (torch.LongTensor): 1D attention_mask for the same example.
            z (torch.FloatTensor): 2D tensor of fused view embeddings (Lz, D).
            vision_feats (torch.FloatTensor): 2D tensor of visual features (Lv, D).
            input_device (torch.device): Device to allocate any new tensors.

        Returns:
            (input_embeds, attn_mask, labels):
                - input_embeds (torch.FloatTensor): Modified input embeddings.
                - attn_mask (torch.LongTensor): Modified attention mask.
                - labels (torch.LongTensor): Label tensor with answer tokens only.
        '''
        # 1) Locate original vision token span
        vision_start_idx, vision_end_idx = self._find_vision_token_range(ids)

        # 2) Split ids and compute embeddings around the vision region
        ids_before = ids[:vision_start_idx]
        ids_after = ids[vision_end_idx + 1:]

        embed_before = self.vlm.get_input_embeddings()(ids_before)
        embed_after = self.vlm.get_input_embeddings()(ids_after)
        embed_vs = self.vlm.get_input_embeddings()(ids[vision_start_idx].unsqueeze(0))
        embed_ve = self.vlm.get_input_embeddings()(ids[vision_end_idx].unsqueeze(0))

        # Concatenate: [before, <vs>, z, vision_feats, <ve>, after]
        embed = torch.cat([
            embed_before,
            embed_vs,
            z,
            vision_feats,
            embed_ve,
            embed_after
        ], dim=0)

        # 3) Build new attention mask
        mask_before = mask[:vision_start_idx]
        mask_after = mask[vision_end_idx + 1:]
        mask_vs = mask[vision_start_idx].unsqueeze(0)
        mask_ve = mask[vision_end_idx].unsqueeze(0)

        prompt_mask = torch.ones(
            (z.size(0) + vision_feats.size(0),),
            dtype=torch.long,
            device=input_device
        )
        attn = torch.cat([mask_before, mask_vs, prompt_mask, mask_ve, mask_after])

        # 4) Build label tensor: mask out everything except the final answer region
        label = ids.clone()
        label[:] = -100

        label_vs = torch.full((1,), -100, dtype=torch.long, device=input_device)
        label_ve = torch.full((1,), -100, dtype=torch.long, device=input_device)
        label_z = torch.full(
            (z.size(0) + vision_feats.size(0),),
            -100,
            dtype=torch.long,
            device=input_device
        )

        label = torch.cat([
            label[:vision_start_idx],
            label_vs,
            label_z,
            label_ve,
            label[vision_end_idx + 1:]
        ], dim=0)

        # 5) Restore answer tokens in the new label positions
        start_idx_orig, end_idx_orig = self._find_answer_token_range(ids)

        old_vision_len = vision_end_idx - vision_start_idx + 1
        new_vision_len = z.size(0) + vision_feats.size(0) + 2  # <vs> + z + feats + <ve>
        shift = old_vision_len - new_vision_len

        start_idx = self._map_index_after_vision_replacement(
            start_idx_orig, vision_start_idx, vision_end_idx, shift
        )
        end_idx = self._map_index_after_vision_replacement(
            end_idx_orig, vision_start_idx, vision_end_idx, shift
        )

        if start_idx is not None and end_idx is not None:
            label[start_idx:end_idx] = ids[start_idx_orig:end_idx_orig]

        return embed, attn, label

    def _forward_stage1(self, **kwargs):
        '''
        Forward pass for Stage 1 (single-view learning).

        For each sample:
            - Extract visual features using the VLM's visual backbone.
            - Select the corresponding view name from `task`.
            - Apply view-specific cross-attention (q_view[view] → vision_feats).
            - Replace the original vision token region with the resulting z.
            - Compute cross-entropy loss using the VLM language head.
        '''
        pixel_values = kwargs["pixel_values"]
        messages = kwargs["messages"]
        image_grid_thw = kwargs["image_grid_thw"]
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        task = kwargs["task"]

        input_device = pixel_values.device
        B = len(messages)

        # 1) Extract visual features from the VLM
        vision_feats = self.vlm.visual(pixel_values, grid_thw=image_grid_thw)

        # 2) Compute view-specific prompts via cross-attention
        z_list = []
        for i in range(B):
            view_name = task[i]
            q = self.q_view[view_name]
            z_i, _ = self.cross_attn[view_name](q, vision_feats, vision_feats)
            z_list.append(z_i)
        z_stack = torch.stack(z_list, dim=0)  # (B, N, D) under current assumptions

        # 3) Replace vision tokens with z for each example
        input_embeds, attn_mask, labels_list = [], [], []
        for i in range(B):
            embed, attn, label = self._replace_vision_tokens_stage1(
                input_ids[i],
                attention_mask[i],
                z_stack[i],
                input_device
            )
            input_embeds.append(embed)
            attn_mask.append(attn)
            labels_list.append(label)

        # 4) Pad to batch (batch_first=True)
        input_embeds = torch.nn.utils.rnn.pad_sequence(
            input_embeds, batch_first=True
        )
        attn_mask = torch.nn.utils.rnn.pad_sequence(
            attn_mask, batch_first=True
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True
        )

        # 5) Forward through the frozen VLM language head
        outputs = self.vlm(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            labels=labels,
            return_dict=True
        )
        return outputs.loss, outputs

    def _forward_stage2(self, **kwargs):
        '''
        Forward pass for Stage 2 (multi-view fusion).

        For each sample:
            - Extract visual features using the VLM's visual backbone.
            - For every view in VIEW_NAMES, apply cross-attention (q_view[view] → vision_feats).
            - Concatenate all view-specific z's into a single vector z_i.
            - Replace the original vision token region with [z_i, vision_feats].
            - Forward through the VLM to obtain a fusion-based loss.
        '''
        pixel_values = kwargs["pixel_values"]
        messages = kwargs["messages"]
        image_grid_thw = kwargs["image_grid_thw"]
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]

        input_device = pixel_values.device
        B = len(messages)

        # 1) Extract visual features for the whole batch
        vision_feats = self.vlm.visual(pixel_values, grid_thw=image_grid_thw)

        # 2) For each sample, concatenate all view-specific z vectors
        z_all = []
        for i in range(B):
            z_i_list = []
            for view in self.view_names:
                q = self.q_view[view]
                z, _ = self.cross_attn[view](q, vision_feats, vision_feats)
                z_i_list.append(z.squeeze(0))  # (N, D) per view
            z_i = torch.cat(z_i_list, dim=0)   # (sum_views*N, D)
            z_all.append(z_i)
        z_all = torch.stack(z_all, dim=0)      # (B, sum_views*N, D)

        # 3) Replace vision tokens with [z_all[i], vision_feats] for each example
        input_embeds, attn_mask, labels_list = [], [], []
        for i in range(B):
            embed, attn, label = self._replace_vision_tokens_stage2(
                input_ids[i],
                attention_mask[i],
                z_all[i],
                vision_feats,     # Note: original code passes full vision_feats
                input_device
            )
            input_embeds.append(embed)
            attn_mask.append(attn)
            labels_list.append(label)

        # 4) Pad to batch
        input_embeds = torch.nn.utils.rnn.pad_sequence(
            input_embeds, batch_first=True
        )
        attn_mask = torch.nn.utils.rnn.pad_sequence(
            attn_mask, batch_first=True
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True
        )

        # 5) Forward through the VLM language head
        outputs = self.vlm(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            labels=labels,
            return_dict=True
        )
        return outputs.loss, outputs

    def forward(self, **kwargs):
        '''
        Dispatch the forward pass to Stage 1 or Stage 2 logic.

        Args:
            **kwargs: Keyword arguments containing:
                - pixel_values
                - messages
                - image_grid_thw
                - input_ids
                - attention_mask
                - task (Stage 1 only)

        Returns:
            (loss, outputs): Same structure as returned by the underlying VLM.
        '''
        if self.stage == 'stage1':
            return self._forward_stage1(**kwargs)
        elif self.stage == 'stage2':
            return self._forward_stage2(**kwargs)
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

    @torch.no_grad()
    def _generate_stage1(self, **kwargs):
        '''
        Generation logic for Stage 1 (single-view prompt).

        Builds inputs_embeds + attention_mask by injecting a single view's z
        into the vision token region, then calls `vlm.generate`.
        '''
        pixel_values = kwargs["pixel_values"]
        image_grid_thw = kwargs["image_grid_thw"]
        input_ids = kwargs["input_ids"]
        task = kwargs["task"]
        attention_mask = kwargs["attention_mask"]

        input_device = pixel_values.device

        # 1) Extract visual features and treat them as a "memory" for cross-attention
        vision_feats = self.vlm.visual(pixel_values, grid_thw=image_grid_thw)
        vision_feats = vision_feats.unsqueeze(0)  # Match original code's behavior
        B = len(task)

        full_embeds, full_attn_masks = [], []

        # 2) For each sample, inject the corresponding view's z into the sequence
        for i in range(B):
            view_name = task[i]
            q = self.q_view[view_name]

            # Compute view-specific prompt z
            z, _ = self.cross_attn[view_name](
                q.unsqueeze(0),  # (1, N, D)
                vision_feats,
                vision_feats
            )
            z_embed = z.squeeze(0)  # (N, D)
            ids = input_ids[i]
            mask = attention_mask[i]

            # Replace vision region with z
            embed, attn, _ = self._replace_vision_tokens_stage1(
                ids,
                mask,
                z_embed,
                input_device
            )
            full_embeds.append(embed.unsqueeze(0))
            full_attn_masks.append(attn.unsqueeze(0))

        full_embeds = torch.cat(full_embeds, dim=0)
        full_attn_masks = torch.cat(full_attn_masks, dim=0)

        # 3) Strip internal keywords from kwargs before calling generate
        generate_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in {
                "pixel_values", "messages", "image_grid_thw",
                "input_ids", "task", "attention_mask"
            }
        }

        # 4) Call the underlying VLM's generate
        return self.vlm.generate(
            inputs_embeds=full_embeds,
            attention_mask=full_attn_masks,
            max_new_tokens=64,
            temperature=0.8,
            do_sample=True,
            top_p=1.0,
            **generate_kwargs
        )

    @torch.no_grad()
    def _generate_stage2(self, **kwargs):
        '''
        Generation logic for Stage 2 (multi-view fusion).

        Uses all views in VIEW_NAMES to build a fused prompt z, concatenates it
        with vision_feats, replaces the vision token region, and then calls
        `vlm.generate` for a single example.
        '''
        pixel_values = kwargs["pixel_values"]
        image_grid_thw = kwargs["image_grid_thw"]
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]

        input_device = pixel_values.device

        # 1) Extract visual features for the batch
        vision_feats = self.vlm.visual(pixel_values, grid_thw=image_grid_thw)
        vision_feats = vision_feats.unsqueeze(0)  # Original code behavior

        # 2) Build a fused z from all views (assumes batch size == 1)
        z_i_list = []
        for view in self.view_names:
            q = self.q_view[view]
            z, _ = self.cross_attn[view](q.unsqueeze(0), vision_feats, vision_feats)
            z_i_list.append(z.squeeze(0))
        z_i = torch.cat(z_i_list, dim=0)

        # Take the first element of vision_feats for replacement
        vision_feats_ = vision_feats[0]
        ids = input_ids[0]
        mask = attention_mask[0]

        # 3) Replace vision region with [z_i, vision_feats_]
        embed, attn, _ = self._replace_vision_tokens_stage2(
            ids,
            mask,
            z_i,
            vision_feats_,
            input_device
        )
        full_embeds = embed.unsqueeze(0)
        full_attn_mask = attn.unsqueeze(0)

        # 4) Strip internal keywords and call generate
        generate_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in {
                "pixel_values", "messages", "image_grid_thw",
                "input_ids", "task", "attention_mask"
            }
        }

        return self.vlm.generate(
            inputs_embeds=full_embeds,
            attention_mask=full_attn_mask,
            temperature=0.8,
            do_sample=False,
            **generate_kwargs
        )

    @torch.no_grad()
    def generate(self, **kwargs):
        '''
        Dispatch the generation call to Stage 1 or Stage 2 logic.

        Args:
            **kwargs: Keyword arguments similar to forward() but used for generation.

        Returns:
            torch.LongTensor: Generated token sequences from the underlying VLM.
        '''
        if self.stage == 'stage1':
            return self._generate_stage1(**kwargs)
        elif self.stage == 'stage2':
            return self._generate_stage2(**kwargs)
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

    def gradient_checkpointing_enable(self, **kwargs):
        '''
        Enable gradient checkpointing on the underlying VLM, if supported.

        This is useful when you want to reduce memory usage in training,
        especially with large VLMs.
        '''
        if hasattr(self.vlm, "gradient_checkpointing_enable"):
            self.vlm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        '''
        Disable gradient checkpointing on the underlying VLM, if supported.
        '''
        if hasattr(self.vlm, "gradient_checkpointing_disable"):
            self.vlm.gradient_checkpointing_disable()
            
def build_stage1_model(train_config):
    """
    Build the Stage 1 MASP model.

    Loads a frozen VLM and adds view-specific query prompts + cross-attention
    modules for single-view training.
    """
    model_name = train_config["model_name"]
    view_names = train_config["VIEW_NAMES"]

    base_model, processor = load_vlm_model(model_name)
    spmodel = MASPEmotionModel(
        base_model,
        processor,
        view_names=view_names,
        stage="stage1"
    )
    return spmodel, processor


def build_stage2_model(train_config, base_model=None, processor=None):
    """
    Build the Stage 2 MASP model.

    Optionally reuses the Stage 1 base_model/processor if provided.
    Loads pretrained q_view and cross-attention weights from checkpoint.
    """
    # Load VLM if not reused from Stage 1
    if base_model is None or processor is None:
        base_model, processor = load_vlm_model(train_config["model_name"])

    view_names = train_config["VIEW_NAMES"]
    spmodel = MASPEmotionModel(
        base_model,
        processor,
        view_names=view_names,
        stage="stage2"
    )

    # Load pretrained soft-prompt weights from Stage 1 checkpoint
    ckpt = torch.load(train_config["ckpt_path"], map_location="cpu")

    # Load view-specific soft prompts
    for k in spmodel.q_view:
        spmodel.q_view[k] = ckpt["q_view"][k].to(spmodel.q_view[k].device)

    # Load cross-attention weights
    for k in spmodel.cross_attn:
        spmodel.cross_attn[k].load_state_dict(ckpt["cross_attn"][k])

    return spmodel, processor

def register_embedding_gradient_mask(model, prefix_token_ids, num_special_tokens_in_prefix):
    """
    Register a gradient hook so that only the prefix-token embeddings receive updates.
    """
    # Ensure full embedding matrix is marked as trainable
    model.get_input_embeddings().weight.requires_grad = True

    # Indices of embeddings that should receive gradients
    embeddings_to_update = torch.tensor(
        prefix_token_ids[:num_special_tokens_in_prefix],
        dtype=torch.long
    ).to(model.device)

    def grad_hook(grad):
        # Zero out gradients for all tokens except prefix tokens
        mask = torch.zeros_like(grad)
        mask[embeddings_to_update] = 1.0
        return grad * mask

    # Install the backward hook on the embedding weight matrix
    hook_handle = model.get_input_embeddings().weight.register_hook(grad_hook)

    return hook_handle

def setup_prefix_tokens_and_mask(spmodel, processor, train_config):
    """
    Add trainable prefix soft tokens and register a gradient mask so that
    only prefix token embeddings receive gradients in Stage 2.
    """
    num_prefix = train_config["num_special_tokens_in_prefix"]

    # Create tokens: <|reserved_special_token_0|>, <|reserved_special_token_1|>, ...
    prefix_token_strs = [
        f"<|reserved_special_token_{i}|>" for i in range(num_prefix)
    ]

    # Add new tokens to tokenizer and expand LM embeddings
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": prefix_token_strs}
    )
    spmodel.vlm.resize_token_embeddings(len(processor.tokenizer))

    # Convert token strings → ids
    prefix_token_ids = processor.tokenizer.convert_tokens_to_ids(
        prefix_token_strs
    )

    # Combine into one long prefix string to prepend to input text
    prefix = "".join(prefix_token_strs)

    # Enable gradient updates only for prefix embeddings
    embedding = spmodel.vlm.get_input_embeddings()
    embedding.weight.requires_grad = True
    hook_handle = register_embedding_gradient_mask(
        spmodel.vlm,
        prefix_token_ids,
        num_prefix
    )

    return prefix, hook_handle
