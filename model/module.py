import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from DRO_Grasp.utils.se3_transform import compute_link_pose
from DRO_Grasp.utils.multilateration import multilateration
from DRO_Grasp.utils.func_utils import calculate_depth
from DRO_Grasp.utils.pretrain_utils import dist2weight, infonce_loss, mean_order

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

class WeightedL1Loss(nn.Module):
    def __init__(self, links_pc):
        super(WeightedL1Loss, self).__init__()
        self.weights = self._calculate_weights(links_pc)
        self.links_pc = links_pc
    
    def _calculate_weights(self, links_pc):
        # Create a weight tensor
        total_points = sum(link_pc.shape[0] for link_pc in links_pc.values())
        # print("Total points: ", total_points)
        weights = torch.ones(total_points, device=next(iter(links_pc.values())).device)
        
        global_index = 0
        for link_name, link_pc in links_pc.items():
            num_points = link_pc.shape[0]
            weights[global_index:global_index + num_points] = 1.0 / num_points
            global_index += num_points
            
        return weights
    
    def forward(self, pred, target):
        # Calculate absolute difference
        diff = torch.abs(pred - target)
        
        # Apply weights
        weighted_diff = diff * self.weights.unsqueeze(-1)
        
        # Return mean of weighted differences
        return weighted_diff.sum() / len(self.links_pc) 


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg, network, epoch_idx):
        super().__init__()
        self.cfg = cfg
        self.network = network
        self.epoch_idx = epoch_idx

        self.lr = cfg.lr

        os.makedirs(self.cfg.save_dir, exist_ok=True)

    def ddp_print(self, *args, **kwargs):
        if self.global_rank == 0:
            print(*args, **kwargs)

    def training_step(self, batch, batch_idx, visualize=False):
        robot_names = batch['robot_name']  
        object_names = batch['object_name']

        mano_mask = [robot_name == 'mano' for robot_name in robot_names]
        mano_mask = torch.tensor(mano_mask)
        cmap_mask = [robot_name != 'mano' for robot_name in robot_names]
        cmap_mask = torch.tensor(cmap_mask)
        retarget_mask = [object_name.split('+')[0] == 'oakink' for object_name in object_names]
        retarget_mask = torch.tensor(retarget_mask)

        intent = batch['intent'] if 'intent' in batch else None

        robot_links_pc = batch['robot_links_pc']
        robot_pc_initial = batch['robot_pc_initial']
        robot_pc_target = batch['robot_pc_target']

        object_pc = batch['object_pc']
        dro_gt = batch['dro_gt']

        if 'contact_map' in batch and isinstance(batch['contact_map'], torch.Tensor):
            contact_map = batch['contact_map'].unsqueeze(-1)
            object_contact_pc = torch.cat((object_pc, contact_map), dim=-1)

        language_embedding = batch['complex_language_embedding']

        network_output = self.network(
            robot_pc_initial,
            object_contact_pc,
            target_pc = robot_pc_target,
            intent = intent,
            language_emb = language_embedding
        )

        dro = network_output['dro']
        mu = network_output['mu']
        logvar = network_output['logvar']

        mlat_pc = multilateration(dro, object_pc)
        loss = 0.

        if self.cfg.loss_kl:
            loss_kl = - 0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
            loss_kl = torch.sqrt(1 + loss_kl ** 2) - 1
            loss_kl = loss_kl * self.cfg.loss_kl_weight
            self.log('loss_kl', loss_kl, prog_bar=True)
            loss += loss_kl

        if self.cfg.loss_r:
            if self.cfg.loss_r_weighted:
                loss_r = WeightedL1Loss(robot_links_pc[0])(dro, dro_gt)
            else:
                loss_r = nn.L1Loss()(dro, dro_gt)
            loss_r = loss_r * self.cfg.loss_r_weight
            self.log('loss_r', loss_r, prog_bar=True)
            loss += loss_r

        if cmap_mask.sum() > 0:

            cmap_robot_pc_target = robot_pc_target[cmap_mask, :]
            cmap_robot_links_pc = [robot_links_pc[idx] for idx in range(len(robot_links_pc)) if cmap_mask[idx]]
            cmap_mlat_pc = mlat_pc[cmap_mask, :]

            cmap_transformed_pc = None
            cmap_transforms, cmap_transformed_pc = compute_link_pose(cmap_robot_links_pc, cmap_mlat_pc, only_palm=self.cfg.only_palm)

            loss_dict_cmap = {}

            if self.cfg.loss_se3:
                transforms_gt, transformed_pc_gt = compute_link_pose(cmap_robot_links_pc, cmap_robot_pc_target, only_palm=self.cfg.only_palm)
                loss_se3_cmap = 0.0
                for idx in range(len(cmap_transforms)): 
                    transform = cmap_transforms[idx]
                    transform_gt = transforms_gt[idx]
                    loss_se3_item = 0.0
                    for link_idx, link_name in enumerate(transform):
                        rel_translation = transform[link_name][:3, 3] - transform_gt[link_name][:3, 3]
                        rel_rotation = torch.matmul(transform[link_name][:3, :3].T, transform_gt[link_name][:3, :3])
                        rel_rotation_trace = torch.clamp(torch.trace(rel_rotation), -1.0, 3.0)
                        rel_angle = torch.acos((rel_rotation_trace - 1) / 2)
                        if  link_idx != 0 and self.cfg.only_palm:
                            break
                        loss_se3_item += (torch.norm(rel_translation) + rel_angle)
                    loss_se3_cmap += loss_se3_item / len(transform)
                loss_se3_cmap = (loss_se3_cmap / len(cmap_transforms)) * self.cfg.loss_se3_weight
                loss += loss_se3_cmap
                loss_dict_cmap['loss_se3'] = loss_se3_cmap

            if self.cfg.loss_depth:

                cmap_object_name = [object_name for idx, object_name in enumerate(batch['object_name']) if (cmap_mask[idx] and not retarget_mask[idx])]

                if cmap_object_name != []:
                    cmap_not_retarget_transformed_pc = cmap_transformed_pc[(cmap_mask & ~retarget_mask), :]
                    loss_depth_cmap = calculate_depth(cmap_not_retarget_transformed_pc, cmap_object_name)
                    loss_depth_cmap = loss_depth_cmap * self.cfg.loss_depth_weight
                    loss += loss_depth_cmap
                    loss_dict_cmap['loss_depth'] = loss_depth_cmap
                else:
                    loss_dict_cmap['loss_depth'] = 0.0

                if retarget_mask.sum() > 0:
                    assert mano_mask.sum() == 0, "Mano and retarget_shadowhand cannot be in the same batch"
                    retarget_object_name = [object_name for idx, object_name in enumerate(batch['object_name']) if (cmap_mask[idx] and  retarget_mask[idx])]
                    retarget_object_id = [object_id for idx, object_id in enumerate(batch['object_id']) if (cmap_mask[idx] and  retarget_mask[idx])]
                    cmap_retarget_transformed_pc = cmap_transformed_pc[(cmap_mask & retarget_mask), :]
                    loss_depth_retarget = calculate_depth(cmap_retarget_transformed_pc, retarget_object_name, retarget_object_id)
                    loss_depth_retarget = loss_depth_retarget * self.cfg.loss_depth_weight
                    loss += loss_depth_retarget
                    loss_dict_cmap['loss_depth'] += loss_depth_retarget

            for key, value in loss_dict_cmap.items():
                self.log(f'{key}', value, prog_bar=True, on_step=True, on_epoch=True)

        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # def training_step(self, batch, batch_idx):

    #     robot_name = batch['robot_name']

    #     if robot_name[0] == 'mano':
    #         object_id = batch['object_id']
    #     else:
    #         object_id = None

    #     object_name = batch['object_name']

    #     robot_links_pc = batch['robot_links_pc']
    #     robot_pc_initial = batch['robot_pc_initial']
    #     robot_pc_target = batch['robot_pc_target']
    #     object_pc = batch['object_pc']
    #     dro_gt = batch['dro_gt']

    #     network_output = self.network(
    #         robot_pc_initial,
    #         object_pc,
    #         robot_pc_target
    #     )

    #     dro = network_output['dro']
    #     mu = network_output['mu']
    #     logvar = network_output['logvar']

    #     mlat_pc = multilateration(dro, object_pc)

    #     if robot_links_pc is not None:
    #         transforms, transformed_pc = compute_link_pose(robot_links_pc, mlat_pc)

    #     loss = 0.

    #     if self.cfg.loss_kl:
    #         loss_kl = - 0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    #         loss_kl = torch.sqrt(1 + loss_kl ** 2) - 1
    #         loss_kl = loss_kl * self.cfg.loss_kl_weight
    #         self.log('loss_kl', loss_kl, prog_bar=True)
    #         loss += loss_kl

    #     if self.cfg.loss_r:
    #         loss_r = nn.L1Loss()(dro, dro_gt)
    #         loss_r = loss_r * self.cfg.loss_r_weight
    #         self.log('loss_r', loss_r, prog_bar=True)
    #         loss += loss_r

    #     if self.cfg.loss_se3 and robot_links_pc is not None:
    #         transforms_gt, transformed_pc_gt = compute_link_pose(robot_links_pc, robot_pc_target)
    #         loss_se3 = 0.
    #         for idx in range(len(transforms)):  # iteration over batch
    #             transform = transforms[idx]
    #             transform_gt = transforms_gt[idx]
    #             loss_se3_item = 0.
    #             for link_name in transform:
    #                 rel_translation = transform[link_name][:3, 3] - transform_gt[link_name][:3, 3]
    #                 rel_rotation = transform[link_name][:3, :3].mT @ transform_gt[link_name][:3, :3]
    #                 rel_rotation_trace = torch.clamp(torch.trace(rel_rotation), -1, 3)
    #                 rel_angle = torch.acos((rel_rotation_trace - 1) / 2)
    #                 loss_se3_item += torch.mean(torch.norm(rel_translation, dim=-1) + rel_angle)
    #             loss_se3 += loss_se3_item / len(transform)
    #         loss_se3 = loss_se3 / len(transforms) * self.cfg.loss_se3_weight
    #         self.log('loss_se3', loss_se3, prog_bar=True)
    #         loss += loss_se3

    #     if self.cfg.loss_depth and robot_links_pc is not None:
    #         loss_depth = calculate_depth(transformed_pc, object_name, object_id)
    #         loss_depth = loss_depth * self.cfg.loss_depth_weight
    #         self.log('loss_depth', loss_depth, prog_bar=True)
    #         loss += loss_depth

    #     self.log("loss", loss, prog_bar=True)
    #     return loss

    def on_after_backward(self):
        """
        For unknown reasons, there is a small chance that the gradients in CVAE may become NaN during backpropagation.
        In such cases, skip the iteration.
        """
        for param in self.network.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                param.grad = None

    def on_train_epoch_end(self):
        self.epoch_idx += 1
        self.ddp_print(f"Training epoch: {self.epoch_idx}")
        if self.epoch_idx % self.cfg.save_every_n_epoch == 0:
            self.ddp_print(f"Saving state_dict at epoch: {self.epoch_idx}")
            torch.save(self.network.state_dict(), f'{self.cfg.save_dir}/epoch_{self.epoch_idx}.pth')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class PretrainingModule(pl.LightningModule):
    def __init__(self, cfg, encoder):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder

        self.lr = cfg.lr
        self.temperature = cfg.temperature

        self.epoch_idx = 0
        os.makedirs(self.cfg.save_dir, exist_ok=True)

    def ddp_print(self, *args, **kwargs):
        if self.global_rank == 0:
            print(*args, **kwargs)

    def training_step(self, batch, batch_idx, visualize=False):
        robot_pc_1 = batch['robot_pc_1']
        robot_pc_2 = batch['robot_pc_2']

        robot_pc_1 = robot_pc_1 - robot_pc_1.mean(dim=1, keepdims=True)
        robot_pc_2 = robot_pc_2 - robot_pc_2.mean(dim=1, keepdims=True)

        phi_1 = self.encoder(robot_pc_1)  # (B, N, 3) -> (B, N, D)
        phi_2 = self.encoder(robot_pc_2)  # (B, N, 3) -> (B, N, D)

        weights = dist2weight(robot_pc_1, func=lambda x: torch.tanh(10 * x))
        loss, similarity = infonce_loss(
            phi_1, phi_2, weights=weights, temperature=self.temperature
        )
        mean_order_error = mean_order(similarity)

        # loss_nofix, similarity_nofix = infonce_loss(
        #     phi_1, phi_nofix, weights=weights, temperature=self.temperature
        # )
        # mean_order_error_nofix = mean_order(similarity_nofix)

        self.log("mean_order", mean_order_error)
        self.log("loss", loss, prog_bar=True)

        # self.log("mean_order_nofix", mean_order_error_nofix)
        # self.log("loss_nofix", loss_nofix, prog_bar=True)

        # return loss + loss_nofix
        return loss

    def on_train_epoch_end(self):
        self.epoch_idx += 1
        self.ddp_print(f"Training epoch: {self.epoch_idx}")
        if self.epoch_idx % self.cfg.save_every_n_epoch == 0:
            self.ddp_print(f"Saving state_dict at epoch: {self.epoch_idx}")
            torch.save(self.encoder.state_dict(), f'{self.cfg.save_dir}/epoch_{self.epoch_idx}.pth')


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class ThreeDoFModule(pl.LightningModule):

    def __init__(self, cfg, network):
        super().__init__()
        self.cfg = cfg
        self.network = network
        self.lr = cfg.lr
        self.epoch_idx = 0
        os.makedirs(self.cfg.save_dir, exist_ok=True)

    def training_step(self, batch, batch_idx):
        object_pc = batch['object_pc']
        language_emb = batch['language_embedding']
        target_q = [q[3:6] for q in batch['target_q']]
        target_q = torch.stack(target_q)
   
        pose = self.network(object_pc, language_emb)
        loss = nn.MSELoss()(pose, target_q)
        self.log("loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def ddp_print(self, *args, **kwargs):
        if self.global_rank == 0:
            print(*args, **kwargs)
    
    def on_train_epoch_end(self):
        self.epoch_idx += 1
        self.ddp_print(f"Training epoch: {self.epoch_idx}")
        if self.epoch_idx % self.cfg.save_every_n_epoch == 0:
            self.ddp_print(f"Saving state_dict at epoch: {self.epoch_idx}")
            torch.save(self.network.state_dict(), f'{self.cfg.save_dir}/epoch_{self.epoch_idx}.pth')

