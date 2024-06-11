import math
import torch

from torch import nn
from Tool import query_ball_point, index_points


class Point_Embedding(nn.Module):
    def __init__(self, point_embedding_type, init_dim, dims, radius=None, query_num=None):
        super(Point_Embedding, self).__init__()
        self.types = point_embedding_type
        if self.types == "MLP-based":
            self.embedding = nn.Sequential(
                nn.Linear(init_dim, dims),
                nn.LayerNorm(dims),
                nn.ReLU(inplace=True)
            )
        elif self.types == "Semantic-based":
            self.radius = radius
            self.query_num = query_num
            self.embedding = nn.Sequential(
                nn.Linear(init_dim, dims),
                nn.LayerNorm(dims),
                nn.ReLU(inplace=True)
            )
            self.sem_trans = nn.Sequential(
                nn.Linear(dims * 2, int(dims)),
                nn.LayerNorm(int(dims)),
                nn.ReLU(inplace=True)
            )

    def forward(self, inputs, coordinates, distance_matrices):
        if self.types == "MLP-based":
            return self.embedding(inputs)
        elif self.types == "Semantic-based":
            x = self.embedding(inputs)
            query_ball = query_ball_point(self.radius, self.query_num, coordinates, coordinates, distance_matrices)
            ball_features = index_points(x, query_ball)
            relative_position = ball_features - x[:, :, None, ]
            combination = torch.cat([relative_position, x[:, :, None].repeat(1, 1, relative_position.shape[-2], 1)], dim=-1)
            outputs = torch.max(self.sem_trans(combination), dim=-2).values
            return outputs


class Scalar_Attention(nn.Module):
    def __init__(self, dims, position_encode_type, voxel_size, quant_size):
        super(Scalar_Attention, self).__init__()
        self.dims = dims
        self.voxel_size = voxel_size
        self.quant_size = quant_size
        self.position_encode_type = position_encode_type

        self.qkv = nn.Linear(dims, 3 * dims)

        if self.position_encode_type == "Contextual-based":
            quant_length = int((2 * self.voxel_size + 1e-4) // quant_size)
            self.table_x = nn.Parameter(torch.zeros(3, quant_length, dims))  # 0, 1, 2 for q, k, v's x off-set respectively
            self.table_y = nn.Parameter(torch.zeros(3, quant_length, dims))
            self.table_z = nn.Parameter(torch.zeros(3, quant_length, dims))
            nn.init.trunc_normal_(self.table_x, 0.02), nn.init.trunc_normal_(self.table_y, 0.02), nn.init.trunc_normal_(self.table_z, 0.02)

    def forward(self, features, voxel_distances):
        points_num, dims = features.shape
        qkv = self.qkv(features).reshape(points_num, 3, dims).permute(1, 0, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.position_encode_type == "Contextual-based":
            indexes = ((voxel_distances + self.voxel_size) // self.quant_size).clamp(max=self.voxel_size * 2.0 / self.quant_size - 1).int()
            # encode_q = self.table_x[0][indexes[:, :, 0]] + self.table_y[0][indexes[:, :, 1]] + self.table_z[0][indexes[:, :, 2]]
            encode_k = self.table_x[1][indexes[:, :, 0]] + self.table_y[1][indexes[:, :, 1]] + self.table_z[1][indexes[:, :, 2]]
            # encode_v = self.table_x[2][indexes[:, :, 0]] + self.table_y[2][indexes[:, :, 1]] + self.table_z[2][indexes[:, :, 2]]
            # position_bias_query = torch.matmul(q[:, None, ], encode_q.transpose(2, 1)).squeeze(dim=-2)
            position_bias_key = torch.matmul(k[:, None, ], encode_k.transpose(2, 1)).squeeze(dim=-2)
            position_bias = position_bias_key
            attention_map = torch.softmax((torch.matmul(q, k.T) + position_bias) / math.sqrt(self.dims), dim=-1)
            v = v[None]
            attention = torch.matmul(attention_map[:, None, ], v).squeeze(dim=-2)
            return attention

        attention_map = torch.softmax(torch.matmul(q, k.T) / math.sqrt(self.dims), dim=-1)
        attention = torch.matmul(attention_map, v)

        return attention


class Voxel_Encoder(nn.Module):
    def __init__(self, dims, scalar_attention_num, position_encode_type, voxel_size, quant_size):
        super(Voxel_Encoder, self).__init__()
        self.voxel_size = voxel_size

        self.scalar_attention = nn.ModuleList([
            Scalar_Attention(dims, position_encode_type, voxel_size, quant_size) for _ in range(scalar_attention_num)
        ])

        self.trans = nn.Sequential(
            nn.Linear(dims * (scalar_attention_num + 1), dims),
            nn.LayerNorm(dims),
            nn.ReLU(inplace=True)
        )

        self.table = {k: idx for k, idx in enumerate([(m, n, t) for m in range(int(2 / voxel_size)) for n in range(int(2 / voxel_size)) for t in range(int(2 / voxel_size))])}

    def forward(self, inputs, coordinates, groups, effective_groups):
        batch, points_num, dims = inputs.shape
        outputs = []
        for_retrieve = []
        for th in range(batch):
            features, coordinate, group, eff_group = inputs[th], coordinates[th], groups[th], effective_groups[th]
            output = torch.zeros([int(2 / self.voxel_size), int(2 / self.voxel_size), int(2 / self.voxel_size), dims]).to(inputs.device)
            for_ret = torch.zeros([features.shape[0], dims]).to(features.device)
            for i in eff_group:
                voxel = group[i]
                voxel_features = features[voxel]
                x = voxel_features
                voxel_distances = coordinate[voxel][:, None] - coordinate[voxel][None]
                for module in self.scalar_attention:
                    voxel_features = module(voxel_features, voxel_distances)
                    x = torch.cat([x, voxel_features], dim=-1)
                x = torch.max(self.trans(x), dim=0).values
                output[self.table[i][2], self.table[i][1], self.table[i][0], :] = x
                for_ret[voxel] = x
            outputs.append(output)
            for_retrieve.append(for_ret)

        return torch.stack(outputs, dim=0).reshape(batch, -1, dims), torch.stack(for_retrieve, dim=0)


class Point_Voxel_Block(nn.Module):
    def __init__(self, dims):
        super(Point_Voxel_Block, self).__init__()
        self.dims = dims

        self.query = nn.Linear(dims, dims)
        self.key_value = nn.Linear(dims, dims)

        self.trans = nn.Sequential(
            nn.Linear(dims, dims),
            nn.LayerNorm(dims),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        point_feature_set, voxel_feature_set = inputs
        q = self.query(point_feature_set)
        k_v = self.key_value(voxel_feature_set)
        attention_map = torch.matmul(q, k_v.transpose(2, 1)) / math.sqrt(self.dims)
        attention = torch.matmul(attention_map, k_v)
        offset_attention = self.trans(attention - point_feature_set) + point_feature_set

        return offset_attention, voxel_feature_set


class Point_Voxel_Attention(nn.Module):
    def __init__(self, dims, offset_attention_num):
        super(Point_Voxel_Attention, self).__init__()
        self.dims = dims

        self.point_voxel_blocks = nn.Sequential(
            *[Point_Voxel_Block(dims) for _ in range(offset_attention_num)]
        )

    def forward(self, point_feature_set, voxel_feature_set):
        point_features = self.point_voxel_blocks((point_feature_set, voxel_feature_set))[0]
        return point_features


class Locality_Compensation(nn.Module):
    def __init__(self, dims, num_category):
        super(Locality_Compensation, self).__init__()

        self.trans = nn.Sequential(
            nn.Linear(2 * dims + num_category, dims),
            nn.LayerNorm(dims),
            nn.ReLU(inplace=True)
        )

    def forward(self, point_features, compensation_features, classes):
        return self.trans(torch.cat((point_features, compensation_features, classes[:, None, ].repeat(1, point_features.shape[1], 1)), dim=-1))


class Part_Seg_Head(nn.Module):
    def __init__(self, num_part, dims, drop_rate):
        super(Part_Seg_Head, self).__init__()
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(dims, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_part)
        )

    def forward(self, inputs):
        return self.head(inputs)


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class Voxel_Former_Part_Seg(nn.Module):
    def __init__(self, point_embedding_type, init_dim, dims, radius, query_num, scalar_attention_num, position_encode_type, voxel_size, quant_size, offset_attention_num, num_category,
                 drop_rate, num_part):
        super(Voxel_Former_Part_Seg, self).__init__()
        self.point_embedding = Point_Embedding(point_embedding_type=point_embedding_type, init_dim=init_dim, dims=dims, radius=radius, query_num=query_num)
        self.voxel_encoder = Voxel_Encoder(dims=dims, scalar_attention_num=scalar_attention_num, position_encode_type=position_encode_type, voxel_size=voxel_size, quant_size=quant_size)
        self.point_voxel_attention = Point_Voxel_Attention(dims=dims, offset_attention_num=offset_attention_num)
        self.locality_compensation = Locality_Compensation(dims=dims, num_category=num_category)
        self.part_seg_head = Part_Seg_Head(num_part=num_part, dims=dims, drop_rate=drop_rate)

        self.apply(_init_vit_weights)

    def forward(self, inputs, coordinates, distance_matrices, groups, effective_groups, classes):
        point_embedding = self.point_embedding(inputs, coordinates, distance_matrices)
        voxel_encoder, for_retrieve = self.voxel_encoder(point_embedding, coordinates, groups, effective_groups)
        point_voxel_attention = self.point_voxel_attention(point_embedding, voxel_encoder)
        locality_compensation = self.locality_compensation(point_voxel_attention, for_retrieve, classes)
        part_seg_res = self.part_seg_head(locality_compensation)

        return part_seg_res


def get_part_seg(init_dim, dims, voxel_size, query_num=50, scalar_attention_num=4, point_embedding_type="Semantic-based", position_encode_type="Contextual-based", quant_size=0.02,
                 offset_attention_num=4, num_part=50, num_category=16, drop_rate=0.5):

    radius = voxel_size / 2

    return Voxel_Former_Part_Seg(point_embedding_type, init_dim, dims, radius, query_num, scalar_attention_num, position_encode_type, voxel_size, quant_size, offset_attention_num,
                                 num_category, drop_rate, num_part)
