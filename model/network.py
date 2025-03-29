import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gorilla.config import Config
from model.encoder import Encoder, CvaeEncoder
from model.transformer import Transformer
from model.latent_encoder import LatentEncoder
from model.mlp import MLPKernel, MLPKernelLarge
# from OpenAD.utils.model_builder import build_model
from model.encoder import create_encoder_network, create_encoder_network_acc


# def create_openad_pn_encoder_network(pretrain=None, device=torch.device('cpu')) -> nn.Module:
#     cfg = Config.fromfile("/data/zwq/code/OpenAD/config/openad_pn2/full_shape_cfg_downsample.py")
#     encoder = build_model(cfg)
#     if pretrain is not None:
#         print(f"******** Load embedding openad network pretrain from <{pretrain}> ********")
#         encoder.load_state_dict(
#             torch.load(
#                 os.path.join(ROOT_DIR, f"ckpt/pretrain/{pretrain}"),
#                 map_location=device
#             )
#         )
#     return encoder

# def create_openad_dgcnn_encoder_network(pretrain=None, device=torch.device('cpu')) -> nn.Module:
#     cfg = Config.fromfile("/data/zwq/code/OpenAD/config/openad_dgcnn/full_shape_cfg_downsample_512.py")
#     encoder = build_model(cfg)

#     if pretrain is not None:
#         print(f"******** Load embedding openad network pretrain from <{pretrain}> ********")
#         encoder.load_state_dict(
#             torch.load(
#                 os.path.join(ROOT_DIR, f"ckpt/pretrain/{pretrain}"),
#                 map_location=device
#             )
#         )
#     return encoder

def visualize_point_clouds(point_clouds, labels=None, colors=None, title="Point Clouds"):
    """
    Visualizes multiple 3D point clouds in a single plot.
    
    Args:
        point_clouds (list[torch.Tensor or np.ndarray]): List of point clouds, each with shape (N, 3).
        labels (list[str]): List of labels for each point cloud.
        colors (list[str]): List of colors for each point cloud.
        title (str): Title of the plot.
    """
    if labels is None:
        labels = [f"PointCloud {i}" for i in range(len(point_clouds))]
    if colors is None:
        colors = ['r', 'g', 'b', 'c', 'm', 'y']  # Default colors for up to 6 point clouds

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, pc in enumerate(point_clouds):
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().detach().numpy()
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colors[i % len(colors)], s=1, label=labels[i])

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

class ThreeDoFNetwork(nn.Module):
    def __init__(self, cfg, mode):
        super(ThreeDoFNetwork, self).__init__()   
        self.cfg = cfg
        self.mode = mode

        # Debug only
        ckpt = torch.load("/data/zwq/code/DRO-Grasp/output/model_retarget_cross/state_dict/epoch_50.pth")
        for k in list(ckpt.keys()):
            if 'encoder_object.' in k:
                ckpt[k.replace('encoder_object.', '')] = ckpt.pop(k)
            else:
                ckpt.pop(k)

        self.encoder_object = create_encoder_network(emb_dim=cfg.emb_dim)
        self.encoder_object.load_state_dict(ckpt)

        self.fusion_dim = cfg.emb_dim + cfg.language_dim

        self.language_fc = nn.Linear(768, cfg.language_dim)
        nn.init.xavier_uniform_(self.language_fc.weight)
        if self.language_fc.bias is not None:
            nn.init.zeros_(self.language_fc.bias)

        self.global_pool = nn.AdaptiveMaxPool1d(1) 
        self.pose_predictor = nn.Sequential(
            nn.Linear(self.fusion_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  
        )

    def forward(self, object_pc, clip_language_emb):

        object_embedding = self.encoder_object(object_pc)
        object_embedding = object_embedding.detach()

        global_object_embedding = self.global_pool(object_embedding).squeeze()
        language_emb = self.language_fc(clip_language_emb)
        fusion_embedding = torch.cat([global_object_embedding, language_emb], dim=-1)
        pose = self.pose_predictor(fusion_embedding)

        return pose
    
class Network_clip_dgcnn_acc(nn.Module):
    def __init__(self, cfg, mode):
        super(Network_clip_dgcnn_acc, self).__init__()
        self.cfg = cfg
        self.mode = mode

        self.encoder_robot = create_encoder_network(emb_dim=cfg.emb_dim, pretrain=cfg.pretrain)
        self.encoder_object = create_encoder_network_acc(emb_dim=cfg.emb_dim)

        self.transformer_robot = Transformer(emb_dim=cfg.emb_dim+cfg.latent_dim, ff_dims=2*(cfg.emb_dim+cfg.latent_dim), n_blocks=2)
        self.transformer_object = Transformer(emb_dim=cfg.emb_dim+cfg.latent_dim, ff_dims=2*(cfg.emb_dim+cfg.latent_dim), n_blocks=2)

        if cfg.use_intent:
            self.intent_embedding = nn.Embedding(cfg.intent_num, cfg.intent_dim)

        # CVAE encoder
        self.point_encoder = CvaeEncoder(emb_dims=cfg.emb_dim, output_channels=2 * cfg.latent_dim, feat_dim=cfg.emb_dim)
        self.latent_encoder = LatentEncoder(in_dim=2*cfg.latent_dim, dim=4*cfg.latent_dim, out_dim=cfg.latent_dim)

        if cfg.use_intent:
            self.kernel = MLPKernel(cfg.emb_dim + cfg.latent_dim + cfg.intent_dim)
        else:
            self.kernel = MLPKernel(cfg.emb_dim + cfg.latent_dim)
            # self.kernel = MLPKernel(cfg.emb_dim)

    def forward(self, robot_pc, object_pc, target_pc=None, intent=None, 
                visualize=False, cross_object_pc=None, language_emb=None, debug_pc=None):

        if self.cfg.use_intent:
            assert intent is not None, 'Intent is required when use_intent is True.'

        if self.cfg.use_language:
            assert language_emb is not None, 'Language embedding is required when use_language is True.'

        assert not self.cfg.use_intent or not self.cfg.use_language, 'Cannot use both intent and language embedding.'

        if self.cfg.center_pc:  # zero-mean the robot point cloud
            robot_pc = robot_pc - robot_pc.mean(dim=1, keepdim=True)

        # point cloud encoder
        robot_embedding = self.encoder_robot(robot_pc)
        object_embedding = self.encoder_object(object_pc)

        if self.cfg.pretrain is not None:
            robot_embedding = robot_embedding.detach()

        assert self.cfg.use_language, 'Must use language embedding.'
        # CVAE encoder
        if self.cfg.use_language:
            z = language_emb
            mu, logvar = None, None
            
        elif self.mode == 'train' or (self.mode == 'validate' and not self.cfg.use_intent and target_pc is not None): 
            if cross_object_pc is not None:
                grasp_pc = torch.cat([target_pc, cross_object_pc], dim=1)
            else:
                grasp_pc = torch.cat([target_pc, object_pc], dim=1)
            
            grasp_emb = torch.cat([robot_embedding, object_embedding], dim=1)
            latent = self.point_encoder(torch.cat([grasp_pc, grasp_emb], -1))
            mu, logvar = self.latent_encoder(latent)
            z_dist = torch.distributions.normal.Normal(mu, torch.exp(0.5 * logvar))
            z = z_dist.rsample()  # (B, latent_dim)

        else:
            mu, logvar = None, None
            z = torch.randn(robot_pc.shape[0], self.cfg.latent_dim).to(robot_pc.device)
        
        if self.cfg.use_intent:
            intent_embedding = self.intent_embedding(intent).squeeze(1)  
            z = torch.cat([z, intent_embedding], dim=-1)   # (B, latent_dim + intent_dim)


        z = z.unsqueeze(dim=1).repeat(1, robot_embedding.shape[1], 1)  # (B, N, latent_dim + intent_dim)
        
        robot_embedding_z = torch.cat([robot_embedding, z], dim=-1)
        object_embedding_z = torch.cat([object_embedding, z], dim=-1)

        # point cloud transformer
        transformer_robot_outputs = self.transformer_robot(robot_embedding_z, object_embedding_z)
        transformer_object_outputs = self.transformer_object(object_embedding_z, robot_embedding_z)
        robot_embedding_tf = robot_embedding_z + transformer_robot_outputs["src_embedding"]
        object_embedding_tf = object_embedding_z + transformer_object_outputs["src_embedding"]


        Phi_A = robot_embedding_tf
        Phi_B = object_embedding_tf

        # Compute D(R,O) matrix
        if self.cfg.block_computing:  # use matrix block computation to save GPU memory
            B, N, D = Phi_A.shape
            block_num = 4  # experimental result, reaching a balance between speed and GPU memory
            N_block = N // block_num
            assert N % N_block == 0, 'Unable to perform block computation.'

            dro = torch.zeros([B, N, N], dtype=torch.float32, device=Phi_A.device)
            for A_i in range(block_num):
                Phi_A_block = Phi_A[:, A_i * N_block: (A_i + 1) * N_block, :]  # (B, N_block, D)
                for B_i in range(block_num):
                    Phi_B_block = Phi_B[:, B_i * N_block: (B_i + 1) * N_block, :]  # (B, N_block, D)

                    Phi_A_r = Phi_A_block.unsqueeze(2).repeat(1, 1, N_block, 1).reshape(B * N_block * N_block, D)
                    Phi_B_r = Phi_B_block.unsqueeze(1).repeat(1, N_block, 1, 1).reshape(B * N_block * N_block, D)

                    dro[:, A_i * N_block: (A_i + 1) * N_block, B_i * N_block: (B_i + 1) * N_block] \
                        = self.kernel(Phi_A_r, Phi_B_r).reshape(B, N_block, N_block)
        else:
            Phi_A_r = (
                Phi_A.unsqueeze(2)
                .repeat(1, 1, Phi_B.shape[1], 1)
                .reshape(Phi_A.shape[0] * Phi_A.shape[1] * Phi_B.shape[1], Phi_A.shape[2])
            )
            Phi_B_r = (
                Phi_B.unsqueeze(1)
                .repeat(1, Phi_A.shape[1], 1, 1)
                .reshape(Phi_B.shape[0] * Phi_A.shape[1] * Phi_B.shape[1], Phi_B.shape[2])
            )
            dro = self.kernel(Phi_A_r, Phi_B_r).reshape(Phi_A.shape[0], Phi_A.shape[1], Phi_B.shape[1])

        outputs = {
            'dro': dro,
            'mu': mu,
            'logvar': logvar,
        }

         # Visualize
        if visualize:
            point_clouds = [object_pc[0], robot_pc[0]]
            labels = ["Object Point Cloud", "Robot Point Cloud"]
            colors = ['g', 'r']

            if target_pc is not None:
                point_clouds.append(target_pc[0])
                labels.append("Target Robot Point Cloud")
                colors.append('b')
            
        return outputs


def create_network_larger_transformer_clip_dgcnn_acc(cfg, mode):
    network = Network_clip_dgcnn_acc(
        cfg=cfg,
        mode=mode
    )
    return network


class Network_clip_dgcnn(nn.Module):
    def __init__(self, cfg, mode):
        super(Network_clip_dgcnn, self).__init__()
        self.cfg = cfg
        self.mode = mode

        self.encoder_robot = create_encoder_network(emb_dim=cfg.emb_dim, pretrain=cfg.pretrain)
        self.encoder_object = create_encoder_network(emb_dim=cfg.emb_dim)

        self.transformer_robot = Transformer(emb_dim=cfg.emb_dim+cfg.latent_dim, ff_dims=2*(cfg.emb_dim+cfg.latent_dim), n_blocks=2)
        self.transformer_object = Transformer(emb_dim=cfg.emb_dim+cfg.latent_dim, ff_dims=2*(cfg.emb_dim+cfg.latent_dim), n_blocks=2)

        if cfg.use_intent:
            self.intent_embedding = nn.Embedding(cfg.intent_num, cfg.intent_dim)

        # CVAE encoder
        self.point_encoder = CvaeEncoder(emb_dims=cfg.emb_dim, output_channels=2 * cfg.latent_dim, feat_dim=cfg.emb_dim)
        self.latent_encoder = LatentEncoder(in_dim=2*cfg.latent_dim, dim=4*cfg.latent_dim, out_dim=cfg.latent_dim)

        if cfg.use_intent:
            self.kernel = MLPKernel(cfg.emb_dim + cfg.latent_dim + cfg.intent_dim)
        else:
            self.kernel = MLPKernel(cfg.emb_dim + cfg.latent_dim)
            # self.kernel = MLPKernel(cfg.emb_dim)

    def forward(self, robot_pc, object_pc, target_pc=None, intent=None, 
                visualize=False, cross_object_pc=None, language_emb=None, debug_pc=None):

        if self.cfg.use_intent:
            assert intent is not None, 'Intent is required when use_intent is True.'

        if self.cfg.use_language:
            assert language_emb is not None, 'Language embedding is required when use_language is True.'

        assert not self.cfg.use_intent or not self.cfg.use_language, 'Cannot use both intent and language embedding.'

        if self.cfg.center_pc:  # zero-mean the robot point cloud
            robot_pc = robot_pc - robot_pc.mean(dim=1, keepdim=True)

        # point cloud encoder
        robot_embedding = self.encoder_robot(robot_pc)
        object_embedding = self.encoder_object(object_pc)

        if self.cfg.pretrain is not None:
            robot_embedding = robot_embedding.detach()

        assert self.cfg.use_language, 'Must use language embedding.'
        # CVAE encoder
        if self.cfg.use_language:
            z = language_emb
            mu, logvar = None, None
            
        elif self.mode == 'train' or (self.mode == 'validate' and not self.cfg.use_intent and target_pc is not None): 
            if cross_object_pc is not None:
                grasp_pc = torch.cat([target_pc, cross_object_pc], dim=1)
            else:
                grasp_pc = torch.cat([target_pc, object_pc], dim=1)
            
            grasp_emb = torch.cat([robot_embedding, object_embedding], dim=1)
            latent = self.point_encoder(torch.cat([grasp_pc, grasp_emb], -1))
            mu, logvar = self.latent_encoder(latent)
            z_dist = torch.distributions.normal.Normal(mu, torch.exp(0.5 * logvar))
            z = z_dist.rsample()  # (B, latent_dim)

        else:
            mu, logvar = None, None
            z = torch.randn(robot_pc.shape[0], self.cfg.latent_dim).to(robot_pc.device)
        
        if self.cfg.use_intent:
            intent_embedding = self.intent_embedding(intent).squeeze(1)  
            z = torch.cat([z, intent_embedding], dim=-1)   # (B, latent_dim + intent_dim)


        z = z.unsqueeze(dim=1).repeat(1, robot_embedding.shape[1], 1)  # (B, N, latent_dim + intent_dim)
        
        robot_embedding_z = torch.cat([robot_embedding, z], dim=-1)
        object_embedding_z = torch.cat([object_embedding, z], dim=-1)

        # point cloud transformer
        transformer_robot_outputs = self.transformer_robot(robot_embedding_z, object_embedding_z)
        transformer_object_outputs = self.transformer_object(object_embedding_z, robot_embedding_z)
        robot_embedding_tf = robot_embedding_z + transformer_robot_outputs["src_embedding"]
        object_embedding_tf = object_embedding_z + transformer_object_outputs["src_embedding"]

        # Phi_A = torch.cat([robot_embedding_tf, z], dim=-1)  # (B, N, emb_dim + latent_dim + intent_dim)
        # Phi_B = torch.cat([object_embedding_tf, z], dim=-1)  # (B, N, emb_dim + latent_dim + intent_dim)

        Phi_A = robot_embedding_tf
        Phi_B = object_embedding_tf

        # Compute D(R,O) matrix
        if self.cfg.block_computing:  # use matrix block computation to save GPU memory
            B, N, D = Phi_A.shape
            block_num = 4  # experimental result, reaching a balance between speed and GPU memory
            N_block = N // block_num
            assert N % N_block == 0, 'Unable to perform block computation.'

            dro = torch.zeros([B, N, N], dtype=torch.float32, device=Phi_A.device)
            for A_i in range(block_num):
                Phi_A_block = Phi_A[:, A_i * N_block: (A_i + 1) * N_block, :]  # (B, N_block, D)
                for B_i in range(block_num):
                    Phi_B_block = Phi_B[:, B_i * N_block: (B_i + 1) * N_block, :]  # (B, N_block, D)

                    Phi_A_r = Phi_A_block.unsqueeze(2).repeat(1, 1, N_block, 1).reshape(B * N_block * N_block, D)
                    Phi_B_r = Phi_B_block.unsqueeze(1).repeat(1, N_block, 1, 1).reshape(B * N_block * N_block, D)

                    dro[:, A_i * N_block: (A_i + 1) * N_block, B_i * N_block: (B_i + 1) * N_block] \
                        = self.kernel(Phi_A_r, Phi_B_r).reshape(B, N_block, N_block)
        else:
            Phi_A_r = (
                Phi_A.unsqueeze(2)
                .repeat(1, 1, Phi_B.shape[1], 1)
                .reshape(Phi_A.shape[0] * Phi_A.shape[1] * Phi_B.shape[1], Phi_A.shape[2])
            )
            Phi_B_r = (
                Phi_B.unsqueeze(1)
                .repeat(1, Phi_A.shape[1], 1, 1)
                .reshape(Phi_B.shape[0] * Phi_A.shape[1] * Phi_B.shape[1], Phi_B.shape[2])
            )
            dro = self.kernel(Phi_A_r, Phi_B_r).reshape(Phi_A.shape[0], Phi_A.shape[1], Phi_B.shape[1])

        outputs = {
            'dro': dro,
            'mu': mu,
            'logvar': logvar,
        }

         # Visualize
        if visualize:
            point_clouds = [object_pc[0], robot_pc[0]]
            labels = ["Object Point Cloud", "Robot Point Cloud"]
            colors = ['g', 'r']

            if target_pc is not None:
                point_clouds.append(target_pc[0])
                labels.append("Target Robot Point Cloud")
                colors.append('b')

            
        return outputs


def create_network_larger_transformer_clip_dgcnn(cfg, mode):
    network = Network_clip_dgcnn(
        cfg=cfg,
        mode=mode
    )
    return network


class Network_clip_dgcnn_add(nn.Module):
    def __init__(self, cfg, mode):
        super(Network_clip_dgcnn_add, self).__init__()
        self.cfg = cfg
        self.mode = mode

        print('emb_dim:', cfg.emb_dim)
        self.encoder_robot = create_encoder_network(emb_dim=cfg.emb_dim, pretrain=cfg.pretrain)
        self.encoder_object = create_encoder_network(emb_dim=cfg.emb_dim)

        self.transformer_robot = Transformer(emb_dim=cfg.emb_dim, ff_dims=2*(cfg.emb_dim), n_blocks=4)
        self.transformer_object = Transformer(emb_dim=cfg.emb_dim, ff_dims=2*(cfg.emb_dim), n_blocks=4)

        if cfg.use_intent:
            self.intent_embedding = nn.Embedding(cfg.intent_num, cfg.intent_dim)

        if not cfg.use_language:
            # CVAE encoder
            self.point_encoder = CvaeEncoder(emb_dims=cfg.emb_dim, output_channels=2*cfg.latent_dim, feat_dim=cfg.emb_dim)
            self.latent_encoder = LatentEncoder(in_dim=2*cfg.latent_dim, dim=4*cfg.latent_dim, out_dim=cfg.latent_dim)

        if cfg.use_intent:
            self.kernel = MLPKernel(cfg.emb_dim + cfg.intent_dim)
        else:
            self.kernel = MLPKernel(cfg.emb_dim)
            # self.kernel = MLPKernel(cfg.emb_dim)

    def forward(self, robot_pc, object_pc, target_pc=None, intent=None, 
                visualize=False, cross_object_pc=None, language_emb=None, debug_pc=None):

        if self.cfg.use_intent:
            assert intent is not None, 'Intent is required when use_intent is True.'

        if self.cfg.use_language:
            assert language_emb is not None, 'Language embedding is required when use_language is True.'

        assert not self.cfg.use_intent or not self.cfg.use_language, 'Cannot use both intent and language embedding.'

        if self.cfg.center_pc:  # zero-mean the robot point cloud
            robot_pc = robot_pc - robot_pc.mean(dim=1, keepdim=True)

        # point cloud encoder
        robot_embedding = self.encoder_robot(robot_pc)
        object_embedding = self.encoder_object(object_pc)

        if self.cfg.pretrain is not None:
            robot_embedding = robot_embedding.detach()

        assert self.cfg.use_language, 'Must use language embedding.'
        # CVAE encoder
        if self.cfg.use_language:
            z = language_emb
            mu, logvar = None, None
            
        elif self.mode == 'train' or (self.mode == 'validate' and not self.cfg.use_intent and target_pc is not None): 
            if cross_object_pc is not None:
                grasp_pc = torch.cat([target_pc, cross_object_pc], dim=1)
            else:
                grasp_pc = torch.cat([target_pc, object_pc], dim=1)
            
            grasp_emb = torch.cat([robot_embedding, object_embedding], dim=1)
            latent = self.point_encoder(torch.cat([grasp_pc, grasp_emb], -1))
            mu, logvar = self.latent_encoder(latent)
            z_dist = torch.distributions.normal.Normal(mu, torch.exp(0.5 * logvar))
            z = z_dist.rsample()  # (B, latent_dim)

        else:
            mu, logvar = None, None
            z = torch.randn(robot_pc.shape[0], self.cfg.latent_dim).to(robot_pc.device)
        
        if self.cfg.use_intent:
            intent_embedding = self.intent_embedding(intent).squeeze(1)  
            z = torch.cat([z, intent_embedding], dim=-1)   # (B, latent_dim + intent_dim)


        z = z.unsqueeze(dim=1).repeat(1, robot_embedding.shape[1], 1)  # (B, N, latent_dim + intent_dim)
        
        # Element-wise addition
        robot_embedding_z = robot_embedding + z
        object_embedding_z = object_embedding + z

        # point cloud transformer
        transformer_robot_outputs = self.transformer_robot(robot_embedding_z, object_embedding_z)
        transformer_object_outputs = self.transformer_object(object_embedding_z, robot_embedding_z)
        robot_embedding_tf = robot_embedding_z + transformer_robot_outputs["src_embedding"]
        object_embedding_tf = object_embedding_z + transformer_object_outputs["src_embedding"]

        Phi_A = robot_embedding_tf
        Phi_B = object_embedding_tf

        # Compute D(R,O) matrix
        if self.cfg.block_computing:  # use matrix block computation to save GPU memory
            B, N, D = Phi_A.shape
            block_num = 4  # experimental result, reaching a balance between speed and GPU memory
            N_block = N // block_num
            assert N % N_block == 0, 'Unable to perform block computation.'

            dro = torch.zeros([B, N, N], dtype=torch.float32, device=Phi_A.device)
            for A_i in range(block_num):
                Phi_A_block = Phi_A[:, A_i * N_block: (A_i + 1) * N_block, :]  # (B, N_block, D)
                for B_i in range(block_num):
                    Phi_B_block = Phi_B[:, B_i * N_block: (B_i + 1) * N_block, :]  # (B, N_block, D)

                    Phi_A_r = Phi_A_block.unsqueeze(2).repeat(1, 1, N_block, 1).reshape(B * N_block * N_block, D)
                    Phi_B_r = Phi_B_block.unsqueeze(1).repeat(1, N_block, 1, 1).reshape(B * N_block * N_block, D)

                    dro[:, A_i * N_block: (A_i + 1) * N_block, B_i * N_block: (B_i + 1) * N_block] \
                        = self.kernel(Phi_A_r, Phi_B_r).reshape(B, N_block, N_block)
        else:
            Phi_A_r = (
                Phi_A.unsqueeze(2)
                .repeat(1, 1, Phi_B.shape[1], 1)
                .reshape(Phi_A.shape[0] * Phi_A.shape[1] * Phi_B.shape[1], Phi_A.shape[2])
            )
            Phi_B_r = (
                Phi_B.unsqueeze(1)
                .repeat(1, Phi_A.shape[1], 1, 1)
                .reshape(Phi_B.shape[0] * Phi_A.shape[1] * Phi_B.shape[1], Phi_B.shape[2])
            )
            dro = self.kernel(Phi_A_r, Phi_B_r).reshape(Phi_A.shape[0], Phi_A.shape[1], Phi_B.shape[1])

        outputs = {
            'dro': dro,
            'mu': mu,
            'logvar': logvar,
        }

         # Visualize
        if visualize:
            point_clouds = [object_pc[0], robot_pc[0]]
            labels = ["Object Point Cloud", "Robot Point Cloud"]
            colors = ['g', 'r']

            if target_pc is not None:
                point_clouds.append(target_pc[0])
                labels.append("Target Robot Point Cloud")
                colors.append('b')

            # if cross_object_pc is not None:
            #     point_clouds.append(cross_object_pc[0])
            #     labels.append("Cross Object Point Cloud")
            #     colors.append('c')

            # if debug_pc is not None:
            #     point_clouds.append(debug_pc[0])
            #     labels.append("Debug Point Cloud")
            #     colors.append('m')
            
            # if point_clouds != []:
            #     visualize_point_clouds(
            #         point_clouds=point_clouds,
            #         labels=labels,
            #         colors=colors,
            #         title="Additional Point Clouds"
            #     )
            
        return outputs


def create_network_larger_transformer_clip_add_dgcnn_acc(cfg, mode):
    network = Network_clip_dgcnn_add(
        cfg=cfg,
        mode=mode
    )
    return network

class Network_clip_add_dgcnn_acc_cvae(nn.Module):
    def __init__(self, cfg, mode):
        super(Network_clip_add_dgcnn_acc_cvae, self).__init__()
        self.cfg = cfg
        self.mode = mode

        print('emb_dim:', cfg.emb_dim)
        self.encoder_robot = create_encoder_network_acc(emb_dim=cfg.emb_dim, pretrain=cfg.pretrain)
        self.encoder_object = create_encoder_network_acc(emb_dim=cfg.emb_dim)

        self.transformer_robot = Transformer(emb_dim=cfg.emb_dim, ff_dims=2*(cfg.emb_dim), n_blocks=4)
        self.transformer_object = Transformer(emb_dim=cfg.emb_dim, ff_dims=2*(cfg.emb_dim), n_blocks=4)

        self.latent_encoder = LatentEncoder(in_dim=cfg.latent_dim, dim=2*cfg.latent_dim, out_dim=cfg.latent_dim)
        self.kernel = MLPKernel(cfg.emb_dim)

    def forward(self, robot_pc, object_pc, target_pc=None, intent=None, 
                visualize=False, cross_object_pc=None, language_emb=None, debug_pc=None):

        if self.cfg.use_intent:
            assert intent is not None, 'Intent is required when use_intent is True.'

        if self.cfg.use_language:
            assert language_emb is not None, 'Language embedding is required when use_language is True.'

        assert not self.cfg.use_intent or not self.cfg.use_language, 'Cannot use both intent and language embedding.'

        if self.cfg.center_pc:  # zero-mean the robot point cloud
            robot_pc = robot_pc - robot_pc.mean(dim=1, keepdim=True)

        # point cloud encoder
        robot_embedding = self.encoder_robot(robot_pc)
        object_embedding = self.encoder_object(object_pc)

        if self.cfg.pretrain is not None:
            robot_embedding = robot_embedding.detach()

        assert self.cfg.use_language, 'Must use language embedding.'
        # CVAE encoder
        if self.mode == 'train':
            z = language_emb
            mu, logvar = self.latent_encoder(z)
            z_dist = torch.distributions.normal.Normal(mu, torch.exp(0.5 * logvar))
            z_cave = z_dist.rsample()  # (B, latent_dim)
        else:
            mu, logvar = None, None
            z_cave = torch.randn(robot_pc.shape[0], self.cfg.latent_dim).to(robot_pc.device)

        z = z.unsqueeze(dim=1).repeat(1, robot_embedding.shape[1], 1)  # (B, N, latent_dim + intent_dim)
        z_cave = z_cave.unsqueeze(dim=1).repeat(1, robot_embedding.shape[1], 1)  # (B, N, latent_dim)
        
        # Element-wise addition
        robot_embedding_z = robot_embedding + z_cave
        object_embedding_z = object_embedding + z

        # point cloud transformer
        transformer_robot_outputs = self.transformer_robot(robot_embedding_z, object_embedding_z)
        transformer_object_outputs = self.transformer_object(object_embedding_z, robot_embedding_z)
        robot_embedding_tf = robot_embedding_z + transformer_robot_outputs["src_embedding"]
        object_embedding_tf = object_embedding_z + transformer_object_outputs["src_embedding"]

        Phi_A = robot_embedding_tf
        Phi_B = object_embedding_tf

        # Compute D(R,O) matrix
        if self.cfg.block_computing:  # use matrix block computation to save GPU memory
            B, N, D = Phi_A.shape
            block_num = 4  # experimental result, reaching a balance between speed and GPU memory
            N_block = N // block_num
            assert N % N_block == 0, 'Unable to perform block computation.'

            dro = torch.zeros([B, N, N], dtype=torch.float32, device=Phi_A.device)
            for A_i in range(block_num):
                Phi_A_block = Phi_A[:, A_i * N_block: (A_i + 1) * N_block, :]  # (B, N_block, D)
                for B_i in range(block_num):
                    Phi_B_block = Phi_B[:, B_i * N_block: (B_i + 1) * N_block, :]  # (B, N_block, D)

                    Phi_A_r = Phi_A_block.unsqueeze(2).repeat(1, 1, N_block, 1).reshape(B * N_block * N_block, D)
                    Phi_B_r = Phi_B_block.unsqueeze(1).repeat(1, N_block, 1, 1).reshape(B * N_block * N_block, D)

                    dro[:, A_i * N_block: (A_i + 1) * N_block, B_i * N_block: (B_i + 1) * N_block] \
                        = self.kernel(Phi_A_r, Phi_B_r).reshape(B, N_block, N_block)
        else:
            Phi_A_r = (
                Phi_A.unsqueeze(2)
                .repeat(1, 1, Phi_B.shape[1], 1)
                .reshape(Phi_A.shape[0] * Phi_A.shape[1] * Phi_B.shape[1], Phi_A.shape[2])
            )
            Phi_B_r = (
                Phi_B.unsqueeze(1)
                .repeat(1, Phi_A.shape[1], 1, 1)
                .reshape(Phi_B.shape[0] * Phi_A.shape[1] * Phi_B.shape[1], Phi_B.shape[2])
            )
            dro = self.kernel(Phi_A_r, Phi_B_r).reshape(Phi_A.shape[0], Phi_A.shape[1], Phi_B.shape[1])

        outputs = {
            'dro': dro,
            'mu': mu,
            'logvar': logvar,
        }

         # Visualize
        if visualize:
            point_clouds = [object_pc[0], robot_pc[0]]
            labels = ["Object Point Cloud", "Robot Point Cloud"]
            colors = ['g', 'r']

            if target_pc is not None:
                point_clouds.append(target_pc[0])
                labels.append("Target Robot Point Cloud")
                colors.append('b')
            
        return outputs


def create_network_larger_transformer_clip_add_dgcnn_acc_cvae(cfg, mode):
    network = Network_clip_add_dgcnn_acc_cvae(
        cfg=cfg,
        mode=mode
    )
    return network

class Network_clip_add_dgcnn_acc(nn.Module):
    def __init__(self, cfg, mode):
        super(Network_clip_add_dgcnn_acc, self).__init__()
        self.cfg = cfg
        self.mode = mode

        print('emb_dim:', cfg.emb_dim)
        self.encoder_robot = create_encoder_network_acc(emb_dim=cfg.emb_dim, pretrain=cfg.pretrain)
        self.encoder_object = create_encoder_network_acc(emb_dim=cfg.emb_dim)

        self.transformer_robot = Transformer(emb_dim=cfg.emb_dim, ff_dims=2*(cfg.emb_dim), n_blocks=4)
        self.transformer_object = Transformer(emb_dim=cfg.emb_dim, ff_dims=2*(cfg.emb_dim), n_blocks=4)

        if cfg.use_intent:
            self.intent_embedding = nn.Embedding(cfg.intent_num, cfg.intent_dim)

        if not cfg.use_language:
            # CVAE encoder
            self.point_encoder = CvaeEncoder(emb_dims=cfg.emb_dim, output_channels=2*cfg.latent_dim, feat_dim=cfg.emb_dim)
            self.latent_encoder = LatentEncoder(in_dim=2*cfg.latent_dim, dim=4*cfg.latent_dim, out_dim=cfg.latent_dim)

        if cfg.use_intent:
            self.kernel = MLPKernel(cfg.emb_dim + cfg.intent_dim)
        else:
            self.kernel = MLPKernel(cfg.emb_dim)

    def forward(self, robot_pc, object_pc, target_pc=None, intent=None, 
                visualize=False, cross_object_pc=None, language_emb=None, debug_pc=None):

        if self.cfg.use_intent:
            assert intent is not None, 'Intent is required when use_intent is True.'

        if self.cfg.use_language:
            assert language_emb is not None, 'Language embedding is required when use_language is True.'

        assert not self.cfg.use_intent or not self.cfg.use_language, 'Cannot use both intent and language embedding.'

        if self.cfg.center_pc:  # zero-mean the robot point cloud
            robot_pc = robot_pc - robot_pc.mean(dim=1, keepdim=True)

        # point cloud encoder
        robot_embedding = self.encoder_robot(robot_pc)
        object_embedding = self.encoder_object(object_pc)

        if self.cfg.pretrain is not None:
            robot_embedding = robot_embedding.detach()

        assert self.cfg.use_language, 'Must use language embedding.'
        # CVAE encoder
        if self.cfg.use_language:
            z = language_emb
            mu, logvar = None, None
            
        elif self.mode == 'train' or (self.mode == 'validate' and not self.cfg.use_intent and target_pc is not None): 
            if cross_object_pc is not None:
                grasp_pc = torch.cat([target_pc, cross_object_pc], dim=1)
            else:
                grasp_pc = torch.cat([target_pc, object_pc], dim=1)
            
            grasp_emb = torch.cat([robot_embedding, object_embedding], dim=1)
            latent = self.point_encoder(torch.cat([grasp_pc, grasp_emb], -1))
            mu, logvar = self.latent_encoder(latent)
            z_dist = torch.distributions.normal.Normal(mu, torch.exp(0.5 * logvar))
            z = z_dist.rsample()  # (B, latent_dim)

        else:
            mu, logvar = None, None
            z = torch.randn(robot_pc.shape[0], self.cfg.latent_dim).to(robot_pc.device)
        
        if self.cfg.use_intent:
            intent_embedding = self.intent_embedding(intent).squeeze(1)  
            z = torch.cat([z, intent_embedding], dim=-1)   # (B, latent_dim + intent_dim)


        z = z.unsqueeze(dim=1).repeat(1, robot_embedding.shape[1], 1)  # (B, N, latent_dim + intent_dim)
        
        # Element-wise addition
        robot_embedding_z = robot_embedding + z
        object_embedding_z = object_embedding + z

        # point cloud transformer
        transformer_robot_outputs = self.transformer_robot(robot_embedding_z, object_embedding_z)
        transformer_object_outputs = self.transformer_object(object_embedding_z, robot_embedding_z)
        robot_embedding_tf = robot_embedding_z + transformer_robot_outputs["src_embedding"]
        object_embedding_tf = object_embedding_z + transformer_object_outputs["src_embedding"]

        Phi_A = robot_embedding_tf
        Phi_B = object_embedding_tf

        # Compute D(R,O) matrix
        if self.cfg.block_computing:  # use matrix block computation to save GPU memory
            B, N, D = Phi_A.shape
            block_num = 4  # experimental result, reaching a balance between speed and GPU memory
            N_block = N // block_num
            assert N % N_block == 0, 'Unable to perform block computation.'

            dro = torch.zeros([B, N, N], dtype=torch.float32, device=Phi_A.device)
            for A_i in range(block_num):
                Phi_A_block = Phi_A[:, A_i * N_block: (A_i + 1) * N_block, :]  # (B, N_block, D)
                for B_i in range(block_num):
                    Phi_B_block = Phi_B[:, B_i * N_block: (B_i + 1) * N_block, :]  # (B, N_block, D)

                    Phi_A_r = Phi_A_block.unsqueeze(2).repeat(1, 1, N_block, 1).reshape(B * N_block * N_block, D)
                    Phi_B_r = Phi_B_block.unsqueeze(1).repeat(1, N_block, 1, 1).reshape(B * N_block * N_block, D)

                    dro[:, A_i * N_block: (A_i + 1) * N_block, B_i * N_block: (B_i + 1) * N_block] \
                        = self.kernel(Phi_A_r, Phi_B_r).reshape(B, N_block, N_block)
        else:
            Phi_A_r = (
                Phi_A.unsqueeze(2)
                .repeat(1, 1, Phi_B.shape[1], 1)
                .reshape(Phi_A.shape[0] * Phi_A.shape[1] * Phi_B.shape[1], Phi_A.shape[2])
            )
            Phi_B_r = (
                Phi_B.unsqueeze(1)
                .repeat(1, Phi_A.shape[1], 1, 1)
                .reshape(Phi_B.shape[0] * Phi_A.shape[1] * Phi_B.shape[1], Phi_B.shape[2])
            )
            dro = self.kernel(Phi_A_r, Phi_B_r).reshape(Phi_A.shape[0], Phi_A.shape[1], Phi_B.shape[1])

        outputs = {
            'dro': dro,
            'mu': mu,
            'logvar': logvar,
        }

         # Visualize
        if visualize:
            point_clouds = [object_pc[0], robot_pc[0]]
            labels = ["Object Point Cloud", "Robot Point Cloud"]
            colors = ['g', 'r']

            if target_pc is not None:
                point_clouds.append(target_pc[0])
                labels.append("Target Robot Point Cloud")
                colors.append('b')
            
        return outputs


def create_network_larger_transformer_clip_add_dgcnn_acc(cfg, mode):
    network = Network_clip_add_dgcnn_acc(
        cfg=cfg,
        mode=mode
    )
    return network


class Network_clip_cat_dgcnn_acc(nn.Module):
    def __init__(self, cfg, mode):
        super(Network_clip_cat_dgcnn_acc, self).__init__()
        self.cfg = cfg
        self.mode = mode

        self.encoder_robot = create_encoder_network_acc(emb_dim=cfg.emb_dim, pretrain=cfg.pretrain)
        self.encoder_object = create_encoder_network_acc(emb_dim=cfg.emb_dim)

        self.transformer_robot = Transformer(emb_dim=cfg.emb_dim+cfg.latent_dim, ff_dims=2*(cfg.emb_dim+cfg.latent_dim), n_blocks=2)
        self.transformer_object = Transformer(emb_dim=cfg.emb_dim+cfg.latent_dim, ff_dims=2*(cfg.emb_dim+cfg.latent_dim), n_blocks=2)

        if cfg.use_intent:
            self.intent_embedding = nn.Embedding(cfg.intent_num, cfg.intent_dim)

        # CVAE encoder
        self.point_encoder = CvaeEncoder(emb_dims=cfg.emb_dim, output_channels=2 * cfg.latent_dim, feat_dim=cfg.emb_dim)
        self.latent_encoder = LatentEncoder(in_dim=2*cfg.latent_dim, dim=4*cfg.latent_dim, out_dim=cfg.latent_dim)

        if cfg.use_intent:
            self.kernel = MLPKernel(cfg.emb_dim + cfg.latent_dim + cfg.intent_dim)
        else:
            self.kernel = MLPKernel(cfg.emb_dim + cfg.latent_dim)
            # self.kernel = MLPKernel(cfg.emb_dim)

    def forward(self, robot_pc, object_pc, target_pc=None, intent=None, 
                visualize=False, cross_object_pc=None, language_emb=None, debug_pc=None):

        if self.cfg.use_intent:
            assert intent is not None, 'Intent is required when use_intent is True.'

        if self.cfg.use_language:
            assert language_emb is not None, 'Language embedding is required when use_language is True.'

        assert not self.cfg.use_intent or not self.cfg.use_language, 'Cannot use both intent and language embedding.'

        if self.cfg.center_pc:  # zero-mean the robot point cloud
            robot_pc = robot_pc - robot_pc.mean(dim=1, keepdim=True)

        # point cloud encoder
        robot_embedding = self.encoder_robot(robot_pc)
        object_embedding = self.encoder_object(object_pc)

        if self.cfg.pretrain is not None:
            robot_embedding = robot_embedding.detach()

        assert self.cfg.use_language, 'Must use language embedding.'
        # CVAE encoder
        if self.cfg.use_language:
            z = language_emb
            mu, logvar = None, None
            
        elif self.mode == 'train' or (self.mode == 'validate' and not self.cfg.use_intent and target_pc is not None): 
            if cross_object_pc is not None:
                grasp_pc = torch.cat([target_pc, cross_object_pc], dim=1)
            else:
                grasp_pc = torch.cat([target_pc, object_pc], dim=1)
            
            grasp_emb = torch.cat([robot_embedding, object_embedding], dim=1)
            latent = self.point_encoder(torch.cat([grasp_pc, grasp_emb], -1))
            mu, logvar = self.latent_encoder(latent)
            z_dist = torch.distributions.normal.Normal(mu, torch.exp(0.5 * logvar))
            z = z_dist.rsample()  # (B, latent_dim)

        else:
            mu, logvar = None, None
            z = torch.randn(robot_pc.shape[0], self.cfg.latent_dim).to(robot_pc.device)
        
        if self.cfg.use_intent:
            intent_embedding = self.intent_embedding(intent).squeeze(1)  
            z = torch.cat([z, intent_embedding], dim=-1)   # (B, latent_dim + intent_dim)


        z = z.unsqueeze(dim=1).repeat(1, robot_embedding.shape[1], 1)  # (B, N, latent_dim + intent_dim)
        
        robot_embedding_z = torch.cat([robot_embedding, z], dim=-1)
        object_embedding_z = torch.cat([object_embedding, z], dim=-1)

        # point cloud transformer
        transformer_robot_outputs = self.transformer_robot(robot_embedding_z, object_embedding_z)
        transformer_object_outputs = self.transformer_object(object_embedding_z, robot_embedding_z)
        robot_embedding_tf = robot_embedding_z + transformer_robot_outputs["src_embedding"]
        object_embedding_tf = object_embedding_z + transformer_object_outputs["src_embedding"]

        # Phi_A = torch.cat([robot_embedding_tf, z], dim=-1)  # (B, N, emb_dim + latent_dim + intent_dim)
        # Phi_B = torch.cat([object_embedding_tf, z], dim=-1)  # (B, N, emb_dim + latent_dim + intent_dim)

        Phi_A = robot_embedding_tf
        Phi_B = object_embedding_tf

        # Compute D(R,O) matrix
        if self.cfg.block_computing:  # use matrix block computation to save GPU memory
            B, N, D = Phi_A.shape
            block_num = 4  # experimental result, reaching a balance between speed and GPU memory
            N_block = N // block_num
            assert N % N_block == 0, 'Unable to perform block computation.'

            dro = torch.zeros([B, N, N], dtype=torch.float32, device=Phi_A.device)
            for A_i in range(block_num):
                Phi_A_block = Phi_A[:, A_i * N_block: (A_i + 1) * N_block, :]  # (B, N_block, D)
                for B_i in range(block_num):
                    Phi_B_block = Phi_B[:, B_i * N_block: (B_i + 1) * N_block, :]  # (B, N_block, D)

                    Phi_A_r = Phi_A_block.unsqueeze(2).repeat(1, 1, N_block, 1).reshape(B * N_block * N_block, D)
                    Phi_B_r = Phi_B_block.unsqueeze(1).repeat(1, N_block, 1, 1).reshape(B * N_block * N_block, D)

                    dro[:, A_i * N_block: (A_i + 1) * N_block, B_i * N_block: (B_i + 1) * N_block] \
                        = self.kernel(Phi_A_r, Phi_B_r).reshape(B, N_block, N_block)
        else:
            Phi_A_r = (
                Phi_A.unsqueeze(2)
                .repeat(1, 1, Phi_B.shape[1], 1)
                .reshape(Phi_A.shape[0] * Phi_A.shape[1] * Phi_B.shape[1], Phi_A.shape[2])
            )
            Phi_B_r = (
                Phi_B.unsqueeze(1)
                .repeat(1, Phi_A.shape[1], 1, 1)
                .reshape(Phi_B.shape[0] * Phi_A.shape[1] * Phi_B.shape[1], Phi_B.shape[2])
            )
            dro = self.kernel(Phi_A_r, Phi_B_r).reshape(Phi_A.shape[0], Phi_A.shape[1], Phi_B.shape[1])

        outputs = {
            'dro': dro,
            'mu': mu,
            'logvar': logvar,
        }

         # Visualize
        if visualize:
            point_clouds = [object_pc[0], robot_pc[0]]
            labels = ["Object Point Cloud", "Robot Point Cloud"]
            colors = ['g', 'r']

            if target_pc is not None:
                point_clouds.append(target_pc[0])
                labels.append("Target Robot Point Cloud")
                colors.append('b')

            
        return outputs

def create_network_larger_transformer_clip_cat_dgcnn_acc(cfg, mode):
    network = Network_clip_cat_dgcnn_acc(
        cfg=cfg,
        mode=mode
    )
    return network


class Network_openai_cat_dgcnn_acc(nn.Module):
    def __init__(self, cfg, mode):
        super(Network_openai_cat_dgcnn_acc, self).__init__()
        self.cfg = cfg
        self.mode = mode

        self.encoder_robot = create_encoder_network_acc(emb_dim=cfg.emb_dim, pretrain=cfg.pretrain)
        self.encoder_object = create_encoder_network_acc(emb_dim=cfg.emb_dim)

        self.transformer_robot = Transformer(emb_dim=cfg.emb_dim+cfg.latent_dim, ff_dims=2*(cfg.emb_dim+cfg.latent_dim), n_blocks=2)
        self.transformer_object = Transformer(emb_dim=cfg.emb_dim+cfg.latent_dim, ff_dims=2*(cfg.emb_dim+cfg.latent_dim), n_blocks=2)

        if cfg.use_intent:
            self.intent_embedding = nn.Embedding(cfg.intent_num, cfg.intent_dim)

        # CVAE encoder
        self.point_encoder = CvaeEncoder(emb_dims=cfg.emb_dim, output_channels=2 * cfg.latent_dim, feat_dim=cfg.emb_dim)
        self.latent_encoder = LatentEncoder(in_dim=2*cfg.latent_dim, dim=4*cfg.latent_dim, out_dim=cfg.latent_dim)

        if cfg.use_intent:
            self.kernel = MLPKernel(cfg.emb_dim + cfg.latent_dim + cfg.intent_dim)
        else:
            self.kernel = MLPKernel(cfg.emb_dim + cfg.latent_dim)
            # self.kernel = MLPKernel(cfg.emb_dim)

    def forward(self, robot_pc, object_pc, target_pc=None, intent=None, 
                visualize=False, cross_object_pc=None, language_emb=None, debug_pc=None):

        if self.cfg.use_intent:
            assert intent is not None, 'Intent is required when use_intent is True.'

        if self.cfg.use_language:
            assert language_emb is not None, 'Language embedding is required when use_language is True.'

        assert not self.cfg.use_intent or not self.cfg.use_language, 'Cannot use both intent and language embedding.'

        if self.cfg.center_pc:  # zero-mean the robot point cloud
            robot_pc = robot_pc - robot_pc.mean(dim=1, keepdim=True)

        # point cloud encoder
        robot_embedding = self.encoder_robot(robot_pc)
        object_embedding = self.encoder_object(object_pc)

        if self.cfg.pretrain is not None:
            robot_embedding = robot_embedding.detach()

        assert self.cfg.use_language, 'Must use language embedding.'
        # CVAE encoder
        if self.cfg.use_language:
            z = language_emb
            mu, logvar = None, None
            
        elif self.mode == 'train' or (self.mode == 'validate' and not self.cfg.use_intent and target_pc is not None): 
            if cross_object_pc is not None:
                grasp_pc = torch.cat([target_pc, cross_object_pc], dim=1)
            else:
                grasp_pc = torch.cat([target_pc, object_pc], dim=1)
            
            grasp_emb = torch.cat([robot_embedding, object_embedding], dim=1)
            latent = self.point_encoder(torch.cat([grasp_pc, grasp_emb], -1))
            mu, logvar = self.latent_encoder(latent)
            z_dist = torch.distributions.normal.Normal(mu, torch.exp(0.5 * logvar))
            z = z_dist.rsample()  # (B, latent_dim)

        else:
            mu, logvar = None, None
            z = torch.randn(robot_pc.shape[0], self.cfg.latent_dim).to(robot_pc.device)
        
        if self.cfg.use_intent:
            intent_embedding = self.intent_embedding(intent).squeeze(1)  
            z = torch.cat([z, intent_embedding], dim=-1)   # (B, latent_dim + intent_dim)


        z = z.unsqueeze(dim=1).repeat(1, robot_embedding.shape[1], 1)  # (B, N, latent_dim + intent_dim)
        
        robot_embedding_z = torch.cat([robot_embedding, z], dim=-1)
        object_embedding_z = torch.cat([object_embedding, z], dim=-1)

        # point cloud transformer
        transformer_robot_outputs = self.transformer_robot(robot_embedding_z, object_embedding_z)
        transformer_object_outputs = self.transformer_object(object_embedding_z, robot_embedding_z)
        robot_embedding_tf = robot_embedding_z + transformer_robot_outputs["src_embedding"]
        object_embedding_tf = object_embedding_z + transformer_object_outputs["src_embedding"]

        # Phi_A = torch.cat([robot_embedding_tf, z], dim=-1)  # (B, N, emb_dim + latent_dim + intent_dim)
        # Phi_B = torch.cat([object_embedding_tf, z], dim=-1)  # (B, N, emb_dim + latent_dim + intent_dim)

        Phi_A = robot_embedding_tf
        Phi_B = object_embedding_tf

        # Compute D(R,O) matrix
        if self.cfg.block_computing:  # use matrix block computation to save GPU memory
            B, N, D = Phi_A.shape
            block_num = 4  # experimental result, reaching a balance between speed and GPU memory
            N_block = N // block_num
            assert N % N_block == 0, 'Unable to perform block computation.'

            dro = torch.zeros([B, N, N], dtype=torch.float32, device=Phi_A.device)
            for A_i in range(block_num):
                Phi_A_block = Phi_A[:, A_i * N_block: (A_i + 1) * N_block, :]  # (B, N_block, D)
                for B_i in range(block_num):
                    Phi_B_block = Phi_B[:, B_i * N_block: (B_i + 1) * N_block, :]  # (B, N_block, D)

                    Phi_A_r = Phi_A_block.unsqueeze(2).repeat(1, 1, N_block, 1).reshape(B * N_block * N_block, D)
                    Phi_B_r = Phi_B_block.unsqueeze(1).repeat(1, N_block, 1, 1).reshape(B * N_block * N_block, D)

                    dro[:, A_i * N_block: (A_i + 1) * N_block, B_i * N_block: (B_i + 1) * N_block] \
                        = self.kernel(Phi_A_r, Phi_B_r).reshape(B, N_block, N_block)
        else:
            Phi_A_r = (
                Phi_A.unsqueeze(2)
                .repeat(1, 1, Phi_B.shape[1], 1)
                .reshape(Phi_A.shape[0] * Phi_A.shape[1] * Phi_B.shape[1], Phi_A.shape[2])
            )
            Phi_B_r = (
                Phi_B.unsqueeze(1)
                .repeat(1, Phi_A.shape[1], 1, 1)
                .reshape(Phi_B.shape[0] * Phi_A.shape[1] * Phi_B.shape[1], Phi_B.shape[2])
            )
            dro = self.kernel(Phi_A_r, Phi_B_r).reshape(Phi_A.shape[0], Phi_A.shape[1], Phi_B.shape[1])

        outputs = {
            'dro': dro,
            'mu': mu,
            'logvar': logvar,
        }

         # Visualize
        if visualize:
            point_clouds = [object_pc[0], robot_pc[0]]
            labels = ["Object Point Cloud", "Robot Point Cloud"]
            colors = ['g', 'r']

            if target_pc is not None:
                point_clouds.append(target_pc[0])
                labels.append("Target Robot Point Cloud")
                colors.append('b')

            # if cross_object_pc is not None:
            #     point_clouds.append(cross_object_pc[0])
            #     labels.append("Cross Object Point Cloud")
            #     colors.append('c')

            # if debug_pc is not None:
            #     point_clouds.append(debug_pc[0])
            #     labels.append("Debug Point Cloud")
            #     colors.append('m')
            
            # if point_clouds != []:
            #     visualize_point_clouds(
            #         point_clouds=point_clouds,
            #         labels=labels,
            #         colors=colors,
            #         title="Additional Point Clouds"
            #     )
            
        return outputs


def create_network_openai_cat_dgcnn_acc(cfg, mode):
    network = Network_openai_cat_dgcnn_acc(
        cfg=cfg,
        mode=mode
    )
    return network


class Network_openai_cat_dgcnn(nn.Module):
    def __init__(self, cfg, mode):
        super(Network_openai_cat_dgcnn, self).__init__()
        self.cfg = cfg
        self.mode = mode

        self.encoder_robot = create_encoder_network(emb_dim=cfg.emb_dim, pretrain=cfg.pretrain)
        self.encoder_object = create_encoder_network(emb_dim=cfg.emb_dim)

        self.transformer_robot = Transformer(emb_dim=cfg.emb_dim+cfg.latent_dim, ff_dims=2*(cfg.emb_dim+cfg.latent_dim), n_blocks=2)
        self.transformer_object = Transformer(emb_dim=cfg.emb_dim+cfg.latent_dim, ff_dims=2*(cfg.emb_dim+cfg.latent_dim), n_blocks=2)

        if cfg.use_intent:
            self.intent_embedding = nn.Embedding(cfg.intent_num, cfg.intent_dim)

        # CVAE encoder
        self.point_encoder = CvaeEncoder(emb_dims=cfg.emb_dim, output_channels=2 * cfg.latent_dim, feat_dim=cfg.emb_dim)
        self.latent_encoder = LatentEncoder(in_dim=2*cfg.latent_dim, dim=4*cfg.latent_dim, out_dim=cfg.latent_dim)

        if cfg.use_intent:
            self.kernel = MLPKernel(cfg.emb_dim + cfg.latent_dim + cfg.intent_dim)
        else:
            self.kernel = MLPKernel(cfg.emb_dim + cfg.latent_dim)
            # self.kernel = MLPKernel(cfg.emb_dim)

    def forward(self, robot_pc, object_pc, target_pc=None, intent=None, 
                visualize=False, cross_object_pc=None, language_emb=None, debug_pc=None):

        if self.cfg.use_intent:
            assert intent is not None, 'Intent is required when use_intent is True.'

        if self.cfg.use_language:
            assert language_emb is not None, 'Language embedding is required when use_language is True.'

        assert not self.cfg.use_intent or not self.cfg.use_language, 'Cannot use both intent and language embedding.'

        if self.cfg.center_pc:  # zero-mean the robot point cloud
            robot_pc = robot_pc - robot_pc.mean(dim=1, keepdim=True)

        # point cloud encoder
        robot_embedding = self.encoder_robot(robot_pc)
        object_embedding = self.encoder_object(object_pc)

        if self.cfg.pretrain is not None:
            robot_embedding = robot_embedding.detach()

        object_embedding = object_embedding.detach()

        assert self.cfg.use_language, 'Must use language embedding.'
        # CVAE encoder
        if self.cfg.use_language:
            z = language_emb
            mu, logvar = None, None
            
        elif self.mode == 'train' or (self.mode == 'validate' and not self.cfg.use_intent and target_pc is not None): 
            if cross_object_pc is not None:
                grasp_pc = torch.cat([target_pc, cross_object_pc], dim=1)
            else:
                grasp_pc = torch.cat([target_pc, object_pc], dim=1)
            
            grasp_emb = torch.cat([robot_embedding, object_embedding], dim=1)
            latent = self.point_encoder(torch.cat([grasp_pc, grasp_emb], -1))
            mu, logvar = self.latent_encoder(latent)
            z_dist = torch.distributions.normal.Normal(mu, torch.exp(0.5 * logvar))
            z = z_dist.rsample()  # (B, latent_dim)

        else:
            mu, logvar = None, None
            z = torch.randn(robot_pc.shape[0], self.cfg.latent_dim).to(robot_pc.device)
        
        if self.cfg.use_intent:
            intent_embedding = self.intent_embedding(intent).squeeze(1)  
            z = torch.cat([z, intent_embedding], dim=-1)   # (B, latent_dim + intent_dim)


        z = z.unsqueeze(dim=1).repeat(1, robot_embedding.shape[1], 1)  # (B, N, latent_dim + intent_dim)
        
        robot_embedding_z = torch.cat([robot_embedding, z], dim=-1)
        object_embedding_z = torch.cat([object_embedding, z], dim=-1)

        # point cloud transformer
        transformer_robot_outputs = self.transformer_robot(robot_embedding_z, object_embedding_z)
        transformer_object_outputs = self.transformer_object(object_embedding_z, robot_embedding_z)
        robot_embedding_tf = robot_embedding_z + transformer_robot_outputs["src_embedding"]
        object_embedding_tf = object_embedding_z + transformer_object_outputs["src_embedding"]

        # Phi_A = torch.cat([robot_embedding_tf, z], dim=-1)  # (B, N, emb_dim + latent_dim + intent_dim)
        # Phi_B = torch.cat([object_embedding_tf, z], dim=-1)  # (B, N, emb_dim + latent_dim + intent_dim)

        Phi_A = robot_embedding_tf
        Phi_B = object_embedding_tf

        # Compute D(R,O) matrix
        if self.cfg.block_computing:  # use matrix block computation to save GPU memory
            B, N, D = Phi_A.shape
            block_num = 4  # experimental result, reaching a balance between speed and GPU memory
            N_block = N // block_num
            assert N % N_block == 0, 'Unable to perform block computation.'

            dro = torch.zeros([B, N, N], dtype=torch.float32, device=Phi_A.device)
            for A_i in range(block_num):
                Phi_A_block = Phi_A[:, A_i * N_block: (A_i + 1) * N_block, :]  # (B, N_block, D)
                for B_i in range(block_num):
                    Phi_B_block = Phi_B[:, B_i * N_block: (B_i + 1) * N_block, :]  # (B, N_block, D)

                    Phi_A_r = Phi_A_block.unsqueeze(2).repeat(1, 1, N_block, 1).reshape(B * N_block * N_block, D)
                    Phi_B_r = Phi_B_block.unsqueeze(1).repeat(1, N_block, 1, 1).reshape(B * N_block * N_block, D)

                    dro[:, A_i * N_block: (A_i + 1) * N_block, B_i * N_block: (B_i + 1) * N_block] \
                        = self.kernel(Phi_A_r, Phi_B_r).reshape(B, N_block, N_block)
        else:
            Phi_A_r = (
                Phi_A.unsqueeze(2)
                .repeat(1, 1, Phi_B.shape[1], 1)
                .reshape(Phi_A.shape[0] * Phi_A.shape[1] * Phi_B.shape[1], Phi_A.shape[2])
            )
            Phi_B_r = (
                Phi_B.unsqueeze(1)
                .repeat(1, Phi_A.shape[1], 1, 1)
                .reshape(Phi_B.shape[0] * Phi_A.shape[1] * Phi_B.shape[1], Phi_B.shape[2])
            )
            dro = self.kernel(Phi_A_r, Phi_B_r).reshape(Phi_A.shape[0], Phi_A.shape[1], Phi_B.shape[1])

        outputs = {
            'dro': dro,
            'mu': mu,
            'logvar': logvar,
        }

         # Visualize
        if visualize:
            point_clouds = [object_pc[0], robot_pc[0]]
            labels = ["Object Point Cloud", "Robot Point Cloud"]
            colors = ['g', 'r']

            if target_pc is not None:
                point_clouds.append(target_pc[0])
                labels.append("Target Robot Point Cloud")
                colors.append('b')
            
        return outputs


def create_network_openai_cat_dgcnn(cfg, mode):
    network = Network_openai_cat_dgcnn(
        cfg=cfg,
        mode=mode
    )
    return network


class Network(nn.Module):
    def __init__(self, cfg, mode):
        super(Network, self).__init__()
        self.cfg = cfg
        self.mode = mode

        self.encoder_robot = create_encoder_network(emb_dim=cfg.emb_dim, pretrain=cfg.pretrain)
        self.encoder_object = create_encoder_network(emb_dim=cfg.emb_dim)

        self.transformer_robot = Transformer(emb_dim=cfg.emb_dim)
        self.transformer_object = Transformer(emb_dim=cfg.emb_dim)

        # CVAE encoder
        self.point_encoder = CvaeEncoder(emb_dims=cfg.emb_dim, output_channels=2 * cfg.latent_dim, feat_dim=cfg.emb_dim)
        self.latent_encoder = LatentEncoder(in_dim=2*cfg.latent_dim, dim=4*cfg.latent_dim, out_dim=cfg.latent_dim)

        self.kernel = MLPKernel(cfg.emb_dim + cfg.latent_dim)

    def forward(self, robot_pc, object_pc, target_pc=None):
        if self.cfg.center_pc:  # zero-mean the robot point cloud
            robot_pc = robot_pc - robot_pc.mean(dim=1, keepdim=True)

        # point cloud encoder
        robot_embedding = self.encoder_robot(robot_pc)
        object_embedding = self.encoder_object(object_pc)

        if self.cfg.pretrain is not None:
            robot_embedding = robot_embedding.detach()

        # point cloud transformer
        transformer_robot_outputs = self.transformer_robot(robot_embedding, object_embedding)
        transformer_object_outputs = self.transformer_object(object_embedding, robot_embedding)
        robot_embedding_tf = robot_embedding + transformer_robot_outputs["src_embedding"]
        object_embedding_tf = object_embedding + transformer_object_outputs["src_embedding"]

        # CVAE encoder
        if self.mode == 'train':
            grasp_pc = torch.cat([target_pc, object_pc], dim=1)
            grasp_emb = torch.cat([robot_embedding_tf, object_embedding_tf], dim=1)
            latent = self.point_encoder(torch.cat([grasp_pc, grasp_emb], -1))
            mu, logvar = self.latent_encoder(latent)
            z_dist = torch.distributions.normal.Normal(mu, torch.exp(0.5 * logvar))
            z = z_dist.rsample()  # (B, latent_dim)
        else:
            mu, logvar = None, None
            z = torch.randn(robot_pc.shape[0], self.cfg.latent_dim).to(robot_pc.device)
        z = z.unsqueeze(dim=1).repeat(1, robot_embedding_tf.shape[1], 1)  # (B, N, latent_dim)

        Phi_A = torch.cat([robot_embedding_tf, z], dim=-1)  # (B, N, emb_dim + latent_dim)
        Phi_B = torch.cat([object_embedding_tf, z], dim=-1)  # (B, N, emb_dim + latent_dim)

        # Compute D(R,O) matrix
        if self.cfg.block_computing:  # use matrix block computation to save GPU memory
            B, N, D = Phi_A.shape
            block_num = 4  # experimental result, reaching a balance between speed and GPU memory
            N_block = N // block_num
            assert N % N_block == 0, 'Unable to perform block computation.'

            dro = torch.zeros([B, N, N], dtype=torch.float32, device=Phi_A.device)
            for A_i in range(block_num):
                Phi_A_block = Phi_A[:, A_i * N_block: (A_i + 1) * N_block, :]  # (B, N_block, D)
                for B_i in range(block_num):
                    Phi_B_block = Phi_B[:, B_i * N_block: (B_i + 1) * N_block, :]  # (B, N_block, D)

                    Phi_A_r = Phi_A_block.unsqueeze(2).repeat(1, 1, N_block, 1).reshape(B * N_block * N_block, D)
                    Phi_B_r = Phi_B_block.unsqueeze(1).repeat(1, N_block, 1, 1).reshape(B * N_block * N_block, D)

                    dro[:, A_i * N_block: (A_i + 1) * N_block, B_i * N_block: (B_i + 1) * N_block] \
                        = self.kernel(Phi_A_r, Phi_B_r).reshape(B, N_block, N_block)
        else:
            Phi_A_r = (
                Phi_A.unsqueeze(2)
                .repeat(1, 1, Phi_A.shape[1], 1)
                .reshape(Phi_A.shape[0] * Phi_A.shape[1] * Phi_A.shape[1], Phi_A.shape[2])
            )
            Phi_B_r = (
                Phi_B.unsqueeze(1)
                .repeat(1, Phi_B.shape[1], 1, 1)
                .reshape(Phi_B.shape[0] * Phi_B.shape[1] * Phi_B.shape[1], Phi_B.shape[2])
            )
            dro = self.kernel(Phi_A_r, Phi_B_r).reshape(Phi_A.shape[0], Phi_A.shape[1], Phi_B.shape[1])

        outputs = {
            'dro': dro,
            'mu': mu,
            'logvar': logvar,
        }
        return outputs


def create_network(cfg, mode):
    network = Network(
        cfg=cfg,
        mode=mode
    )
    return network
