import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


def create_encoder_network(emb_dim, pretrain=None, device=torch.device('cpu')) -> nn.Module:
    encoder = Encoder(emb_dim=emb_dim)
    if pretrain is not None:
        print(f"******** Load embedding network pretrain from <{pretrain}> ********")
        encoder.load_state_dict(
            torch.load(
                os.path.join(ROOT_DIR, f"ckpt/pretrain/{pretrain}"),
                map_location=device
            )
        )
    return encoder

def create_encoder_network_acc(emb_dim, pretrain=None, device=torch.device('cpu')) -> nn.Module:
    encoder = DGCNNAcc(
        input_channels=3,
        emb_dims=emb_dim,
    )
    if pretrain is not None:
        print(f"******** Load embedding network pretrain from <{pretrain}> ********")
        encoder.load_state_dict(
            torch.load(
                os.path.join(ROOT_DIR, f"ckpt/pretrain/{pretrain}"),
                map_location=device
            )
        )
    return encoder

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20):
    idx = knn(x, k=k)
    batch_size, num_points, _ = idx.size()
    _, num_dims, _ = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class Encoder(nn.Module):
    """
    The implementation is based on the DGCNN model
    (https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/pytorch/model.py#L88),
    and https://github.com/r-pad/taxpose/blob/0c4298fa0486fd09e63bf24d618a579b66ba0f18/third_party/dcp/model.py#L282.

    Further explanation can be found in Appendix F.1 of https://arxiv.org/pdf/2410.01702.
    """

    def __init__(self, emb_dim=512):
        super(Encoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(1536, emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, N, 3) -> (B, 3, N)
        B, _, N = x.size()

        x = get_graph_feature(x, k=32)  # (B, 6, N, K)

        x = self.conv1(x)  # (B, 64, N, K)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        x = self.conv2(x)  # (B, 64, N, K)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        x = self.conv3(x)  # (B, 128, N, K)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (B, 128, N)

        x = self.conv4(x)  # (B, 256, N, K)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (B, 256, N)

        x = self.conv5(x)  # (B, 512, N, K)
        x5 = x.max(dim=-1, keepdim=False)[0]  # (B, 512, N)

        global_feat = x5.mean(dim=-1, keepdim=True).repeat(1, 1, N)  # (B, 512, 1) -> (B, 512, N)

        x = torch.cat((x1, x2, x3, x4, x5, global_feat), dim=1)  # (B, 1536, N)
        x = self.conv6(x).view(B, -1, N)  # (B, 512, N)

        return x.permute(0, 2, 1)  # (B, D, N) -> (B, N, D)

def knn_acc(x, k):
    # x: (B, C, N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx

def gather_feature(x, idx):
    """
    Gathers features for each point according to idx.
    Args:
      x: (B, C, N)
      idx: (B, N, k)
    Returns:
      gathered: (B, C, N, k)
    """
    B, C, N = x.shape
    k = idx.shape[-1]
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, N, k)
    x_expanded = x.unsqueeze(-1).expand(-1, -1, N, k)        # (B, C, N, k)
    gathered = torch.gather(x_expanded, 2, idx_expanded)      # (B, C, N, k)
    return gathered

class DGCNNAcc(nn.Module):
    def __init__(self, input_channels=3, k=20, emb_dims=512, P=20):
        """
        Args:
          input_channels: number of input channels (e.g., 3 for XYZ)
          k: number of nearest neighbors for the first layer
          emb_dims: output embedding dimensions
          P: progressive enlargement step for neighbor pooling
        """
        super(DGCNNAcc, self).__init__()
        self.k = k
        self.P = P  # enlargement step

        # Layer 1: from input_channels to 64
        self.conv1a = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2)
        )
        self.conv1b = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2)
        )

        # Layer 2: from 64 to 64
        self.conv2a = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2)
        )
        self.conv2b = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2)
        )

        # Layer 3: from 64 to 128
        self.conv3a = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2)
        )
        self.conv3b = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2)
        )

        # Layer 4: from 128 to 256
        self.conv4a = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.2)
        )
        self.conv4b = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.2)
        )

        # Final embedding layer (aggregates concatenated features)
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims * 2, kernel_size=1, bias=False),
            nn.GroupNorm(16, emb_dims * 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(emb_dims * 2, emb_dims, kernel_size=1, bias=False),
            nn.GroupNorm(16, emb_dims),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) # (B, N, C) -> (B, C, N)
        # x: (B, input_channels, N)
        B, C, N = x.shape
        # Compute a shared neighbor pool.
        # For 4 layers, we need a pool of size: k + 3*P
        pool_size = self.k + 3 * self.P
        idx_pool = knn_acc(x, k=pool_size)  # (B, N, pool_size)

        # For each layer, use a progressively larger subset of neighbors:
        idx1 = idx_pool[:, :, :self.k]                # layer 1: k neighbors
        idx2 = idx_pool[:, :, :self.k + self.P]         # layer 2: k + P neighbors
        idx3 = idx_pool[:, :, :self.k + 2 * self.P]       # layer 3: k + 2P neighbors
        idx4 = idx_pool                                # layer 4: full pool (k + 3P)

        # Define a local helper to apply the "shuffled" graph conv layer.
        def point_conv_layer(x_in, conv_a, conv_b, idx):
            # Apply two pointwise convolutions (MLP branches)
            f_a = conv_a(x_in)  # (B, C_out, N)
            f_b = conv_b(x_in)  # (B, C_out, N)
            # Gather neighbor features from f_a using precomputed indices.
            # Resulting shape: (B, C_out, N, k_subset)
            neighbor_feat = gather_feature(f_a, idx)
            # Aggregate neighbors with max pooling.
            aggregated = torch.max(neighbor_feat, dim=-1)[0]  # (B, C_out, N)
            # Combine with the center branch.
            out = aggregated + f_b
            return out

        # Layer 1
        x1 = point_conv_layer(x, self.conv1a, self.conv1b, idx1)       # (B, 64, N)
        # Layer 2
        x2 = point_conv_layer(x1, self.conv2a, self.conv2b, idx2)          # (B, 64, N)
        # Layer 3
        x3 = point_conv_layer(x2, self.conv3a, self.conv3b, idx3)          # (B, 128, N)
        # Layer 4
        x4 = point_conv_layer(x3, self.conv4a, self.conv4b, idx4)          # (B, 256, N)

        # Concatenate features from all layers along the channel dimension.
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 64+64+128+256 = 512, N)
        x_out = self.conv5(x_cat)  # (B, emb_dims, N)

        return x_out.permute(0, 2, 1) # (B, emb_dims, N) -> (B, N, emb_dims)



class CvaeEncoder(nn.Module):
    """
    The implementation is based on the DGCNN model
    (https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/pytorch/model.py#L88).

    The only modification made is to enable the input to include additional features.
    """

    def __init__(self, emb_dims, output_channels, feat_dim=0):
        super(CvaeEncoder, self).__init__()
        self.feat_dim = feat_dim

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6 + feat_dim, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        B, D, N = x.size()
        x_k = get_graph_feature(x[:, :3, :])  # B, 6, N, K
        x_feat = x[:, 3:, :].unsqueeze(-1).repeat(1, 1, 1, 20) if self.feat_dim != 0 else None  # K = 20
        x = torch.cat([x_k, x_feat], dim=1) if self.feat_dim != 0 else x_k  # (B, 6 + feat_dim, N, K)

        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)[..., 0]

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(B, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(B, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x  # (B, output_channels)
