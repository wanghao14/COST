"""
MART model.

References:
    Copyright (c) 2017 Jie Lei
    Licensed under The MIT License, see https://choosealicense.com/licenses/mit/
    @inproceedings{lei2020mart,
        title={MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning},
        author={Lei, Jie and Wang, Liwei and Shen, Yelong and Yu, Dong and Berg, Tamara L and Bansal, Mohit},
        booktitle={ACL},
        year={2020}
    }

    History:
    https://github.com/jayleicn/recurrent-transformer
    Current version 2021 https://github.com/gingsi/coot-videotext
"""
import copy
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf, open_dict

from mart.configs_mart import MartConfig, MartPathConst
from mart.loss_caption import LabelSmoothingLoss
from nntrainer.utils import count_parameters

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# # default infinity (config.inf = 0), works with fp32. this can lead to NaN values in some circumstances
INF = float('inf')

# # this should be "infinite enough" for -INF to give 0 for masked softmax attention values.
# INF = 1e19
# for fp16 need something like 255


def create_mart_model(cfg: MartConfig, vocab_size: int, cache_dir: str = MartPathConst.CACHE_DIR,
                      verbose: bool = True) -> nn.Module:
    """
    Args:
        cfg: Experiment config.
        vocab_size: Vocabulary, calculated in mart as len(train_set.word2idx).
        cache_dir: Cache directory.
        verbose: Print model name and number of parameters.

    Returns:
        MART model.
    """
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.data.max_position_embeddings = cfg.data.max_v_len + cfg.data.max_t_len
        cfg.data.max_position_embeddings_det = cfg.data.max_d_len + cfg.data.max_t_len
        cfg.data.max_position_embeddings_act = cfg.data.max_a_len + cfg.data.max_t_len
        cfg.data.vocab_size = vocab_size
    
    logger.info("Use non-recurrent single sentence model")
    model = NonRecurTransformer(cfg)

    if cfg.model.use_glove:
        if hasattr(model, "embeddings"):
            logger.info("Load GloVe as word embedding")
            model.embeddings.set_pretrained_embedding(torch.from_numpy(torch.load(
                Path(cache_dir) / f"{cfg.data.name}_vocab_glove.pt")).float(), 
                freeze=cfg.model.freeze_glove)
        else:
            logger.warning("This model has no embeddings, cannot load glove vectors into the model")

    # output model properties
    if verbose:
        print(f"Model: {model.__class__.__name__}")
        count_parameters(model, "Main body")
        if hasattr(model, "embeddings") and hasattr(model.embeddings, "word_embeddings"):
            count_parameters(model.embeddings.word_embeddings, "Word embeddings")

    return model


def gelu(x):
    """
    Implementation of the gelu activation function.
        For information: OpenAI GPT"s gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class MultiCateCrossEntropy(nn.Module):
    def __init__(self):
        super(MultiCateCrossEntropy, self).__init__()

    def forward(self, y_true, y_pred):
        # none_zero_indice = labels.sum(dim=-1) != 0
        # y_true = labels[none_zero_indice]
        # y_pred = preds[none_zero_indice]

        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12

        zeros = torch.zeros((y_pred.shape[0], 1), device=y_pred.device)
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

        neg_loss = torch.logsumexp(y_pred_neg, 1)
        pos_loss = torch.logsumexp(y_pred_pos, 1)
        loss = torch.sum(neg_loss) + torch.sum(pos_loss)
        return loss


class Aligner(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg_d = config.data
        cfg_m = config.model
        self.max_v_len = cfg_d.max_v_len
        self.max_t_len = cfg_d.max_t_len
        self.max_d_len = cfg_d.max_d_len
        self.max_a_len = cfg_d.max_a_len
        self.num_attention_heads = cfg_m.num_attention_heads
        self.attention_head_size = int(cfg_m.hidden_size / cfg_m.num_attention_heads)
        self.all_head_size = cfg_m.hidden_size
        self.vid_projection = nn.Linear(cfg_m.hidden_size, cfg_m.hidden_size)
        self.branch_num = cfg_m.branch.num
        self.branch_action = cfg_m.branch.action_input
        self.branch_detect = cfg_m.branch.detect_input
        if self.branch_num == 1:
            self.attention_vid = BertSelfAttention(cfg_m)
            self.hidden_intermediate = BertIntermediate(cfg_m)
        else:
            if self.branch_detect:
                self.det_projection = nn.Linear(cfg_m.hidden_size, cfg_m.hidden_size)
            if self.branch_action:
                self.act_projection = nn.Linear(cfg_m.hidden_size, cfg_m.hidden_size)
        self.dropout = nn.Dropout(cfg_m.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)
    
    def cross_attention(self, hidden1, hidden2, mask1, mask2, max_len_hidden1, max_len_hidden2):
        if max_len_hidden1 == max_len_hidden2:
            atten_mask_1_2 = make_pad_shifted_mask(mask2, max_len_hidden1, self.max_t_len)
            atten_mask_2_1 = make_pad_shifted_mask(mask1, max_len_hidden2, self.max_t_len)
        elif max_len_hidden1 > max_len_hidden2:
            atten_mask_1_2 = make_pad_shifted_mask_case2(mask2, max_len_hidden1, max_len_hidden2, self.max_t_len)
            atten_mask_2_1 = make_pad_shifted_mask(mask1, max_len_hidden2, self.max_t_len, max_len_hidden1-max_len_hidden2)
        else:
            atten_mask_1_2 = make_pad_shifted_mask(mask2, max_len_hidden1, self.max_t_len, max_len_hidden2-max_len_hidden1)
            atten_mask_2_1 = make_pad_shifted_mask_case2(mask1, max_len_hidden2, max_len_hidden1, self.max_t_len)
        atten_mask_1_2 = (1 - atten_mask_1_2) * -10000.
        atten_mask_2_1 = (1 - atten_mask_2_1) * -10000.
        
        atten_score_1_2 = torch.matmul(hidden1, hidden2.transpose(-1, -2)) 
        atten_score_1_2 = atten_score_1_2 / math.sqrt(self.attention_head_size)
        atten_score_2_1 = atten_score_1_2.transpose(-1, -2)
        atten_score_1_2 = atten_score_1_2 + atten_mask_1_2
        atten_score_2_1 = atten_score_2_1 + atten_mask_2_1
        atten_prob_1_2 = nn.Softmax(dim=-1)(atten_score_1_2)
        atten_prob_2_1 = nn.Softmax(dim=-1)(atten_score_2_1)
        context_1_2 = torch.matmul(atten_prob_1_2, hidden2)
        context_2_1 = torch.matmul(atten_prob_2_1, hidden1)
        
        return context_1_2, context_2_1

    def forward(self, video_embeddings, attention_mask, detect_embeddings, attention_mask_det, action_embeddings,
                attention_mask_act):
        
        if self.branch_num == 1:
            shifted_self_mask = make_pad_shifted_mask(attention_mask, self.max_v_len, self.max_t_len)
            attention_out = self.attention_vid(video_embeddings, video_embeddings, video_embeddings, shifted_self_mask)
            context_embedding = self.hidden_intermediate(attention_out)
            return context_embedding, None, None

        vid_state = self.vid_projection(video_embeddings)
        det_state = self.det_projection(detect_embeddings)  
        act_state = self.act_projection(action_embeddings)  
        
        context_v_d, context_d_v = self.cross_attention(vid_state, det_state, attention_mask, attention_mask_det, self.max_v_len, self.max_d_len)
        context_v_a, context_a_v = self.cross_attention(vid_state, act_state, attention_mask, attention_mask_act, self.max_v_len, self.max_a_len)
        context_a_d, context_d_a = self.cross_attention(act_state, det_state, attention_mask_act, attention_mask_det, self.max_a_len, self.max_d_len)

        context_embedding = context_v_d + context_v_a
        context_embedding_det = context_d_v + context_d_a
        context_embedding_act = context_a_v + context_a_d
        
        return context_embedding, context_embedding_det, context_embedding_act


class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(d_model=6, max_len=10, dropout=0)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:
        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway

        mixed_query_layer = self.query(query_states)  # (N, Lq, 768?)
        mixed_key_layer = self.key(key_states)  # (N, L, 768)
        mixed_value_layer = self.value(value_states)  # (N, L, 768)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh) nh=12 dh=64
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # (N, nh, Lq, dh)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (N, Lq, nh, dh)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # (N, Lq, D)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:
            attention_output: (N, L, D)
        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=0):
    """
    Args:
        input_mask: (N, L) with `1` indicates valid bits, `0` indicates pad
        max_v_len: int, the first `max_v_len` is for video and its padding, the length
            of the rest of the bits is `max_t_len`. We have L = `max_v_len` + `max_t_len`.
            Note max_v_len may also include the memory len (M), thus max_v_len += M
        max_t_len: int
        memory_len: int, M
    Returns:

    >>> max_v_len_ = 2
    >>> max_t_len_ = 3
    >>> input_mask_ = torch.randn(2, 5)
    >>> make_pad_shifted_mask(input_mask_, max_v_len_, max_t_len_)[0]
    tensor([[1., 1., 0., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.]])
    """
    bsz, seq_len = input_mask.shape
    assert max_v_len + max_t_len + memory_len == seq_len, f"max_v_len{max_v_len} max_t_len{max_t_len} seq_len{seq_len}"
    shifted_mask = input_mask.new_zeros(bsz, max_v_len + max_t_len, seq_len)  # (N, L, M+L)
    shifted_mask[:, :, :memory_len + max_v_len] = 1
    shifted_mask[:, max_v_len:, memory_len + max_v_len:] =\
        torch.tril(input_mask.new_ones(max_t_len, max_t_len), diagonal=0)
    return shifted_mask


def make_pad_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=0):
    """
    input_mask: (N, L),
    """
    shifted_mask = make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=memory_len)
    # It's correct to use `input_mask.unsqueeze(1)' instead of
    # `torch.bmm(input_mask.unsqueeze(2), input_mask.unsqueeze(1))'
    # since the rest of the bits are still masked in the subsequent processing steps.
    pad_shifted_mask = shifted_mask * input_mask.unsqueeze(1)
    return pad_shifted_mask


def make_pad_shifted_mask_case2(attention_mask, max_v_len1, max_v_len2, max_t_len):
    """
    input_mask: (N, L),
    """
    bsz = attention_mask.shape[0]
    shifted_attention_mask = attention_mask.new_zeros(bsz, max_v_len1 + max_t_len, max_v_len2 + max_t_len)
    shifted_attention_mask[:, :, :max_v_len2] = 1
    shifted_attention_mask[:, max_v_len1:, max_v_len2:] = \
        torch.tril(attention_mask.new_ones(max_t_len, max_t_len), diagonal=0)
    attention_mask = shifted_attention_mask * attention_mask.unsqueeze(1)
    return attention_mask


def make_video_only_mask(input_mask, max_v_len):
    video_only_mask = copy.deepcopy(input_mask)
    video_only_mask[:, max_v_len:] = 0
    return video_only_mask


class BertLayerNoMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.model
        self.config_d = config.data
        self.branch_num = self.config.branch.num
        self.attention = BertAttention(self.config)
        self.hidden_intermediate = BertIntermediate(self.config)
        self.output = BertOutput(self.config)
        if self.branch_num > 1:
            self.hidden_intermediate2 = BertIntermediate(self.config)
            if self.config.branch.detect_input:
                self.attention_det = BertAttention(self.config)
                self.hidden_intermediate_det = BertIntermediate(self.config)
                self.hidden_intermediate_det2 = BertIntermediate(self.config)
                self.output_det = BertOutput(self.config)

            if self.config.branch.action_input:
                self.attention_act = BertAttention(self.config)
                self.hidden_intermediate_act = BertIntermediate(self.config)
                self.hidden_intermediate_act2 = BertIntermediate(self.config)
                self.output_act = BertOutput(self.config)
        self.aligner = Aligner(config)

    def forward(self, hidden_states, attention_mask, hidden_states_det, attention_mask_det,
                hidden_states_act, attention_mask_act, layer_idx=None, input_name=None):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:
        """
        max_v_len, max_t_len, max_v_len_det = self.config_d.max_v_len, self.config_d.max_t_len, self.config_d.max_d_len
        
        # self-attention, need to shift right
        shifted_self_mask = make_pad_shifted_mask(attention_mask, max_v_len, max_t_len)  # (N, L, L)
        attention_output = self.attention(hidden_states, shifted_self_mask)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D) the output after feed forward

        if hidden_states_det is not None:
            shifted_self_mask_det = make_pad_shifted_mask(attention_mask_det, max_v_len_det, max_t_len)
            attention_output_det = self.attention_det(hidden_states_det, shifted_self_mask_det)
            intermediate_output_det = self.hidden_intermediate_det(attention_output_det) # the output after feed forward
        else:
            intermediate_output_det = None

        if hidden_states_act is not None:
            shifted_self_mask_act = make_pad_shifted_mask(attention_mask_act, max_v_len, max_t_len)
            attention_output_act = self.attention_act(hidden_states_act, shifted_self_mask_act)
            intermediate_output_act = self.hidden_intermediate_act(attention_output_act)
        else:
            intermediate_output_act = None

        intermediate_output_align, intermediate_output_det_align, intermediate_output_act_align = self.aligner(
            intermediate_output, attention_mask, intermediate_output_det, attention_mask_det, intermediate_output_act,
            attention_mask_act)
        if self.branch_num > 1:
            intermediate_output = self.hidden_intermediate2(intermediate_output_align)
        else:
            intermediate_output = intermediate_output_align
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        if intermediate_output_det_align is not None:
            intermediate_output_det = self.hidden_intermediate_det2(intermediate_output_det_align)
            layer_output_det = self.output_det(intermediate_output_det, attention_output_det)
        else:
            layer_output_det = None

        if intermediate_output_act_align is not None:
            intermediate_output_act = self.hidden_intermediate_act2(intermediate_output_act_align)
            layer_output_act = self.output_act(intermediate_output_act, attention_output_act)
        else:
            layer_output_act = None

        return layer_output, layer_output_det, layer_output_act


class BertEncoderNoMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayerNoMemory(config) for _ in range(config.model.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, hidden_states_det, attention_mask_det, hidden_states_act=None,
                attention_mask_act=None, input_name=None, output_all_encoded_layers=True):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:

        Returns:
        """
        all_encoder_layers = []
        all_encoder_layers_det = []
        all_encoder_layers_act = []
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states, hidden_states_det, hidden_states_act = layer_module(hidden_states, attention_mask,
                                                                   hidden_states_det, attention_mask_det,
                                                                   hidden_states_act, attention_mask_act,
                                                                   layer_idx, input_name)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_encoder_layers_det.append(hidden_states_det)
                all_encoder_layers_act.append(hidden_states_act)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_encoder_layers_det.append(hidden_states_det)
            all_encoder_layers_act.append(hidden_states_act)
        return all_encoder_layers, all_encoder_layers_det, all_encoder_layers_act


class BertEmbeddingsWithVideo(nn.Module):
    """
    Construct the embeddings from word (+ video), position and token_type embeddings.
    input_ids (batch_size, sequence_length), with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    ==> video features and word embeddings are merged together by summing up.
    """

    def __init__(self, config):
        super().__init__()
        """
        add_postion_embeddings: whether to add absolute positional embeddings
        """
        cfg_d = config.data
        cfg_m = config.model
        self.max_v_len = cfg_d.max_v_len
        self.max_t_len = cfg_d.max_t_len
        self.max_d_len = cfg_d.max_d_len
        self.add_postion_embeddings = cfg_m.add_postion_embeddings
        self.add_postion_embeddings_det = cfg_m.add_postion_embeddings_det
        self.add_postion_embeddings_act = cfg_m.add_postion_embeddings_act
        self.branch_num = cfg_m.branch.num
        self.branch_detect = cfg_m.branch.detect_input
        self.branch_action = cfg_m.branch.action_input
        self.intermediate_size = cfg_m.intermediate_size
        self.word_embeddings = nn.Embedding(cfg_d.vocab_size, cfg_d.word_vec_size, padding_idx=0)
        self.word_fc = nn.Sequential(
            BertLayerNorm(cfg_d.word_vec_size, eps=cfg_m.layer_norm_eps),
            nn.Dropout(cfg_m.hidden_dropout_prob),
            nn.Linear(cfg_d.word_vec_size, cfg_m.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(cfg_m.hidden_size, eps=cfg_m.layer_norm_eps),
        )

        if self.add_postion_embeddings:
            self.position_embeddings = PositionEncoding(n_filters=cfg_m.hidden_size,
                                                        max_len=cfg_d.max_position_embeddings)
        if self.branch_detect and self.add_postion_embeddings_det:
            self.position_embeddings_det = PositionEncoding(n_filters=cfg_m.hidden_size,
                                                            max_len=cfg_d.max_position_embeddings_det)
        if self.branch_action and self.add_postion_embeddings_act:
            self.position_embeddings_act = PositionEncoding(n_filters=cfg_m.hidden_size,
                                                            max_len=cfg_d.max_position_embeddings_act)
        self.token_type_embeddings = nn.Embedding(cfg_d.type_vocab_size+1, cfg_m.hidden_size)

        if self.branch_num > 2:
            self.video_embeddings = nn.Sequential(
                BertLayerNorm(cfg_d.video_feature_size, eps=cfg_m.layer_norm_eps),
                nn.Dropout(cfg_m.hidden_dropout_prob),
                nn.Linear(cfg_d.video_feature_size, cfg_m.hidden_size),
                nn.ReLU(True),
                BertLayerNorm(cfg_m.hidden_size, eps=cfg_m.layer_norm_eps),
            )
            self.action_embeddings = nn.Sequential(                     # now we use 1024 for the dimension of flow features
                BertLayerNorm(cfg_d.action_feature_size, eps=cfg_m.layer_norm_eps),
                nn.Dropout(cfg_m.hidden_dropout_prob),
                nn.Linear(cfg_d.action_feature_size, cfg_m.hidden_size),
                nn.ReLU(True),
                BertLayerNorm(cfg_m.hidden_size, eps=cfg_m.layer_norm_eps),
            )
            self.LayerNorm_Act = BertLayerNorm(cfg_m.hidden_size, eps=cfg_m.layer_norm_eps)
            self.detect_embeddings = nn.Sequential(
                BertLayerNorm(cfg_d.detect_feature_size, eps=cfg_m.layer_norm_eps),
                nn.Dropout(cfg_m.hidden_dropout_prob),
                nn.Linear(cfg_d.detect_feature_size, cfg_m.hidden_size),
                nn.ReLU(True),
                BertLayerNorm(cfg_m.hidden_size, eps=cfg_m.layer_norm_eps),
            )
            self.LayerNorm_Det = BertLayerNorm(cfg_m.hidden_size, eps=cfg_m.layer_norm_eps)
        elif self.branch_num == 2:
            assert self.branch_detect ^ self.branch_action, "can't input both detection feature and action feature when" \
                                                            "num of branch is only 2"

            if self.branch_detect:
                self.detect_embeddings = nn.Sequential(
                    BertLayerNorm(cfg_d.detect_feature_size, eps=cfg_m.layer_norm_eps),
                    nn.Dropout(cfg_m.hidden_dropout_prob),
                    nn.Linear(cfg_d.detect_feature_size, cfg_m.hidden_size),
                    nn.ReLU(True),
                    BertLayerNorm(cfg_m.hidden_size, eps=cfg_m.layer_norm_eps),
                )
                self.LayerNorm_Det = BertLayerNorm(cfg_m.hidden_size, eps=cfg_m.layer_norm_eps)
                
                self.video_embeddings = nn.Sequential(
                BertLayerNorm(cfg_d.video_feature_size+cfg_d.action_feature_size, eps=cfg_m.layer_norm_eps),
                nn.Dropout(cfg_m.hidden_dropout_prob),
                nn.Linear(cfg_d.video_feature_size+cfg_d.action_feature_size, cfg_m.hidden_size),
                nn.ReLU(True),
                BertLayerNorm(cfg_m.hidden_size, eps=cfg_m.layer_norm_eps),
                )

            if self.branch_action:
                self.action_embeddings = nn.Sequential(  # now we use 1024 for the dimension of flow features
                    BertLayerNorm(cfg_d.action_feature_size, eps=cfg_m.layer_norm_eps),
                    nn.Dropout(cfg_m.hidden_dropout_prob),
                    nn.Linear(cfg_d.action_feature_size, cfg_m.hidden_size),
                    nn.ReLU(True),
                    BertLayerNorm(cfg_m.hidden_size, eps=cfg_m.layer_norm_eps),
                )
                self.LayerNorm_Act = BertLayerNorm(cfg_m.hidden_size, eps=cfg_m.layer_norm_eps)
                
                self.video_embeddings = nn.Sequential(
                BertLayerNorm(cfg_d.video_feature_size, eps=cfg_m.layer_norm_eps),
                nn.Dropout(cfg_m.hidden_dropout_prob),
                nn.Linear(cfg_d.video_feature_size, cfg_m.hidden_size),
                nn.ReLU(True),
                BertLayerNorm(cfg_m.hidden_size, eps=cfg_m.layer_norm_eps),
                )
        elif self.branch_num == 1:
                ori_feat_dim = cfg_d.video_feature_size
                if self.branch_action:
                    ori_feat_dim += cfg_d.action_feature_size
                if self.branch_detect:
                    ori_feat_dim += cfg_d.detect_feature_size * 5
                self.video_embeddings = nn.Sequential(
                    BertLayerNorm(ori_feat_dim, eps=cfg_m.layer_norm_eps),
                    nn.Dropout(cfg_m.hidden_dropout_prob),
                    nn.Linear(ori_feat_dim, cfg_m.hidden_size),
                    nn.ReLU(True),
                    BertLayerNorm(cfg_m.hidden_size, eps=cfg_m.layer_norm_eps),
                )

        
        self.LayerNorm = BertLayerNorm(cfg_m.hidden_size, eps=cfg_m.layer_norm_eps)
        self.dropout = nn.Dropout(cfg_m.hidden_dropout_prob)


    def set_pretrained_embedding(self, pretrained_embedding, freeze=True):
        """
        Note the from_pretrained does not work in-place, so you need to assign value to the embedding
        """
        assert pretrained_embedding.shape == self.word_embeddings.weight.shape  # ensure equal shape
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze,
                                                            padding_idx=self.word_embeddings.padding_idx)


    def forward(self, input_ids, video_features, token_type_ids, detect_ids, detect_features, detect_token_type_ids,
                action_ids=None, action_features=None, action_token_type_ids=None):
        """
        Args:
            input_ids: (N, L)
            video_features: (N, L, D)
            token_type_ids: (N, L, D)

        Returns:
        """
        words_embeddings = self.word_fc(self.word_embeddings(input_ids))
        video_embeddings = self.video_embeddings(video_features)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)      # 0 and 1 for video and text respectively
        embeddings = words_embeddings + video_embeddings + token_type_embeddings
        if self.add_postion_embeddings:
            embeddings = self.position_embeddings(embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        if self.branch_num >= 2:
            if self.branch_detect:
                words_embeddings_det = self.word_fc(self.word_embeddings(detect_ids))
                detect_embeddings = self.detect_embeddings(detect_features)
                token_type_embeddings_det = self.token_type_embeddings(detect_token_type_ids)    # 2 and 1 for det and text respectivelys
                embeddings_det = words_embeddings_det + detect_embeddings + token_type_embeddings_det
                if self.add_postion_embeddings_det:
                    embeddings_det = self.position_embeddings_det(embeddings_det)
                embeddings_det = self.LayerNorm_Det(embeddings_det)
                embeddings_det = self.dropout(embeddings_det)
            else:
                embeddings_det = None

            if self.branch_action:
                words_embeddings_act = self.word_fc(self.word_embeddings(action_ids))
                action_embeddings = self.action_embeddings(action_features)
                token_type_embeddings_act = self.token_type_embeddings(action_token_type_ids)
                embeddings_act = words_embeddings_act + action_embeddings + token_type_embeddings_act
                if self.add_postion_embeddings_act:
                    embeddings_act = self.position_embeddings_act(embeddings_act)
                embeddings_act = self.LayerNorm_Act(embeddings_act)
                embeddings_act = self.dropout(embeddings_act)
            else:
                embeddings_act = None
        
        return embeddings, embeddings_det, embeddings_act  # (N, L, D), (N, L2, D), (N, L3, D)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.m1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.m2 = nn.Linear(config.hidden_size, config.num_class_action, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, hidden_states):
        hidden_states = self.relu(self.m1(hidden_states))
        hidden_states = self.m2(hidden_states)
        return hidden_states

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config, use_for_merge=False):
        super().__init__()
        if use_for_merge:
            self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
            self.LayerNorm = BertLayerNorm(config.hidden_size * 2, eps=config.layer_norm_eps)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.transform_act_fn = gelu

    def forward(self, hidden_states):
        """
        (N, L, D)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config.model)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        if config.model.share_wd_cls_weight:
            assert bert_model_embedding_weights is not None,\
                "bert_model_embedding_weights should not be None "\
                "when setting --share_wd_cls_weight flag to be true"
            assert config.model.hidden_size == bert_model_embedding_weights.size(1),\
                "hidden size has be the same as word embedding size when "\
                "sharing word embedding weight and classifier weight"
            self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                     bert_model_embedding_weights.size(0),
                                     bias=False)
            self.decoder.weight = bert_model_embedding_weights
        else:
            logger.info("The output weights are not the same as the input embeddings!")
            self.decoder = nn.Linear(config.model.hidden_size, config.data.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.data.vocab_size))
        if config.model.branch.detect_input:
            self.transform_det = BertPredictionHeadTransform(config.model)
            self.decoder_det = nn.Linear(config.model.hidden_size, config.model.num_class_detect, bias=False)
            self.bias_det = nn.Parameter(torch.zeros(config.model.num_class_detect))
        #
        if config.model.branch.action_input:
            self.transform_act = BertPredictionHeadTransform(config.model)
            self.decoder_act = nn.Linear(config.model.hidden_size, config.model.num_class_act, bias=False)
            self.bias_act = nn.Parameter(torch.zeros(config.model.num_class_act))

    def forward(self, hidden_states, hidden_states_det, hidden_states_act=None):
    # def forward(self, hidden_states):
        """
        (N, L, D)
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias

        if hidden_states_det is not None:
            hidden_states_det = self.transform_det(hidden_states_det)
            hidden_states_det = self.decoder_det(hidden_states_det) + self.bias_det
        else:
            hidden_states_det = None
        #
        if hidden_states_act is not None:
            hidden_states_act = self.transform_act(hidden_states_act)
            hidden_states_act = self.decoder_act(hidden_states_act[:, 0, :]) + self.bias_act
        else:
            hidden_states_act = None

        return hidden_states, hidden_states_det, hidden_states_act


class NonRecurTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.branch_num = self.cfg.model.branch.num
        self.embeddings = BertEmbeddingsWithVideo(cfg)
        self.encoder = BertEncoderNoMemory(cfg)
        decoder_classifier_weight = self.embeddings.word_embeddings.weight\
            if self.cfg.model.share_wd_cls_weight else None
        self.decoder = BertLMPredictionHead(cfg, decoder_classifier_weight)
        self.loss_func = LabelSmoothingLoss(cfg.model.label_smoothing, cfg.data.vocab_size, ignore_index=-1) \
            if self.cfg.model.label_smoothing > 0 else nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_func_cls = LabelSmoothingLoss(cfg.model.label_smoothing, cfg.model.num_class_detect, ignore_index=-1) \
            if self.cfg.model.label_smoothing > 0 else nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_func_act = MultiCateCrossEntropy()

        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.cfg.optim.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, video_features, input_masks, token_type_ids, input_labels, detect_ids, detect_features,
                detect_masks, detect_token_type_ids, detect_cates, action_ids=None, action_features=None, action_masks=None,
                action_token_type_ids=None, action_cates=None):
        """
        Args:
            input_ids: [(N, L)]
            video_features: [(N, L, D_v)] * step_size
            input_masks: [(N, L)] * step_size with 1 indicates valid bits
            token_type_ids: [(N, L)] * step_size, with `0` on the first `max_v_len` bits, `1` on the last `max_t_len`
            input_labels: [(N, L)] * step_size, with `-1` on ignored positions
            action_features: [(N, L2, D_v)] * step_size
        """
        embeddings, embeddings_det, embeddings_act = self.embeddings(input_ids, video_features, token_type_ids,
                                                                     detect_ids, detect_features, detect_token_type_ids,
                                                                     action_ids, action_features, action_token_type_ids)  # (N, L, D)

        encoded_layer_outputs, encoded_layer_outputs_det, encoded_layer_outputs_act = self.encoder(embeddings, input_masks,
                                                embeddings_det, detect_masks, embeddings_act, action_masks, 
                                                input_name=None, output_all_encoded_layers=False)  # both outputs are list
        prediction_scores, prediction_scores_det, prediction_scores_act = self.decoder(encoded_layer_outputs[-1],
                                                encoded_layer_outputs_det[-1], encoded_layer_outputs_act[-1])  # (N, L, vocab_size)


        if input_labels is not None:
            caption_loss = self.loss_func(prediction_scores.view(-1, self.cfg.data.vocab_size), input_labels.view(-1))
            if self.branch_num == 1:
                detect_loss = None
                action_loss = None
            else:
                if self.cfg.model.branch.detect_input:
                    detect_loss = self.cfg.train.cls_weight * self.loss_func_cls(prediction_scores_det.view(-1, self.cfg.model.num_class_detect),
                                                                                detect_cates.view(-1))
                else:
                    detect_loss = None
                if self.cfg.model.branch.action_input:
                    none_zero_indice = action_cates.sum(dim=-1) != 0
                    action_cates = action_cates[none_zero_indice]
                    prediction_scores_act = prediction_scores_act[none_zero_indice]
                    action_loss = self.cfg.train.act_weight * self.loss_func_act(action_cates, prediction_scores_act)
                else:
                    action_loss = None

        else:
            caption_loss = None
            detect_loss = None
            action_loss = None
        return caption_loss, prediction_scores, detect_loss, action_loss
