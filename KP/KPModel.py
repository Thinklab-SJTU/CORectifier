import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from KPEnv import *


class KPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = KP_Encoder(**model_params)
        self.decoder = KP_Decoder(**model_params)
        self.encoded_items = None
        # shape: (batch, item, embedding)

    def pre_forward(self, reset_state: Reset_State):
        self.encoded_items = self.encoder(reset_state.pseudo_points)
        # shape: (batch, item, embedding)
        self.decoder.set_kv(self.encoded_items)

    def forward(
        self, state: Step_State,
        rectify_mask: Tensor = None,
        # act_prob: float = 0.9,
        # act_mask: Tensor = None
    ):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        ref_items = state.ref_items

        if state.current_item is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))
            # shape: (batch, pomo)
            encoded_first_item = _get_encoding(self.encoded_items, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_item)

        else:
            encoded_last_item = _get_encoding(self.encoded_items, state.current_item)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_item, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, job)

            if self.training or self.model_params["eval_type"] == "softmax":
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break

                if rectify_mask is not None:
                    selected: Tensor = selected.reshape(batch_size * pomo_size)
                    cur_item = state.current_item.reshape(batch_size * pomo_size)
                    available = (state.ninf_mask == 0).reshape(batch_size * pomo_size, -1)
                    # shape: (batch * pomo, item)
                    ref_items = ref_items.unsqueeze(1).expand(batch_size, pomo_size, ref_items.shape[-1]) \
                        .reshape(batch_size * pomo_size, -1) # omit the dummy item
                    # shape: (batch * pomo, item)
                    selected = _rectify_action(
                        cur_item=cur_item,
                        init_action=selected,
                        # act_prob=act_prob,
                        # act_mask=act_mask,
                        instance_mask=rectify_mask,
                        ref_items=ref_items,
                        available=available
                    ).reshape(batch_size, pomo_size)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    prob = torch.where(prob == 0, prob + 1e-6, prob)

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None

        return selected, prob


def _get_encoding(encoded_items: Tensor, item_index_to_pick: Tensor):
    # encoded_items.shape: (batch, problem, embedding)
    # item_index_to_pick.shape: (batch, pomo)

    batch_size = item_index_to_pick.size(0)
    pomo_size = item_index_to_pick.size(1)
    embedding_dim = encoded_items.size(2)

    gathering_index = item_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_items = encoded_items.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_items


def _rectify_action(
    cur_item: Tensor,
    init_action: Tensor,
    instance_mask: Tensor,
    ref_items: Tensor,
    available: Tensor
):
    # cur_item.shape: (batch * pomo, )
    # init_action.shape: (batch * pomo, )
    # instance_mask.shape: (batch * pomo, )
    # act_mask.shape: (batch * pomo, )
    # ref_items.shape: (batch * pomo, item)
    # available.shape: (batch * pomo, item+1)
    with torch.no_grad():
        bs, items_num = ref_items.shape
        # 1. 找到所有可用物品的坐标 (行索引, 列索引)
        avail_item_idx = torch.nonzero(ref_items & available[:, :-1])  # [N, 2]

        # 2. 统计每行可用物品数量，并生成每行的随机索引
        row_indices = avail_item_idx[:, 0]  # 所有非零元素的行索引 [N,]
        col_indices = avail_item_idx[:, 1]  # 所有非零元素的列索引 [N,]
        counts = torch.bincount(row_indices, minlength=bs)  # 每行可用数量 [bs,]

        # 3. 生成每行的随机偏移量（在可用列索引中随机选一个）
        # 计算每行可用列索引的起始位置
        offsets = torch.zeros(bs + 1, dtype=torch.long)
        offsets[1:] = counts.cumsum(dim=0)  # [0, c0, c0+c1, ..., total]
        # 为每行生成一个随机索引（相对于该行可用列的偏移）
        # 生成每行的随机索引（修正部分）
        max_count = counts.max().item()  # 取counts的最大值（整数）
        if max_count > 0: # at least one trajectory has not finished
            rand_offsets = torch.randint(0, max_count, (bs,))  # 基础随机数
            valid_mask = counts > 0  # 标记有可用元素的行（counts[i] > 0）
            rand_offsets[valid_mask] = rand_offsets[valid_mask] % counts[valid_mask]
            # 转换为全局索引（在col_indices中的位置）
            global_indices = offsets[:-1] + rand_offsets  # [bs,]

            # 4. 提取每行随机选中的列索引（全零行用dummy值）
            selected_cols = torch.full((bs,), items_num, dtype=torch.long)
            valid_rows = counts > 0  # 标记有可用物品的行
            selected_cols[valid_rows] = col_indices[global_indices[valid_rows]]

            new_action = torch.where(instance_mask, selected_cols, init_action)
        else:
            new_action = init_action

    del avail_item_idx, row_indices, col_indices  # 删除大张量引用
    torch.cuda.empty_cache()

    return new_action


########################################
# ENCODER
########################################

class KP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        encoder_layer_num = self.model_params["encoder_layer_num"]

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data):
        # data.shape: (batch, item, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, item, embedding)

        out = embedded_input
        for layer in self.layers:
            out = layer(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params["head_num"]

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################

class KP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_items):
        # encoded_items.shape: (batch, problem, embedding)
        head_num = self.model_params["head_num"]

        self.k = reshape_by_heads(self.Wk(encoded_items), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_items), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_items.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params["head_num"]

        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_item: Tensor, ninf_mask):
        # encoded_last_item.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params["head_num"]

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_last_item), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = self.q_first + q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params["sqrt_embedding_dim"]
        logit_clipping = self.model_params["logit_clipping"]

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv: Tensor, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(
        q: Tensor, k: Tensor, v: Tensor,
        rank2_ninf_mask=None, rank3_ninf_mask=None
    ):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
