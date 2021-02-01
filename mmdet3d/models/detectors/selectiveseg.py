import torch
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector
from mmcv.runner import force_fp32

from selective_seg.cylinder_encoder import Cylinder_VFE

import ipdb

def xyz2polar(pts_xyz):
    rho = torch.sqrt(pts_xyz[:, 0] ** 2 + pts_xyz[:, 1] ** 2)
    phi = torch.atan2(pts_xyz[:, 1], pts_xyz[:, 0])
    return torch.stack((rho, phi, pts_xyz[:, 2]), dim=1)

def polar2xyz(polar):
    x = polar[:, 0] * torch.cos(polar[:, 1])
    y = polar[:, 0] * torch.sin(polar[:, 1])
    return torch.stack((x, y, polar[:, 2]), dim=-1)


@DETECTORS.register_module()
class SelectiveSeg(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 cylinder_voxel_encoder=None,
                 cylinder_pillar_encoder=None,
                 pred_mode=['voxel']):
        super(SelectiveSeg,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)

        self.voxel_encoder = Cylinder_VFE(**cylinder_voxel_encoder)
        self.pillar_encoder = Cylinder_VFE(**cylinder_pillar_encoder)

        self.pred_mode = pred_mode

        self.global_idx = 0

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, return_feat=False):
        output_shape = self.pts_voxel_cfg['output_shape']
        cylinder_range = torch.tensor(self.pts_voxel_cfg['cylinder_range']).type_as(points[0])
        voxel_size = (cylinder_range[3:] - cylinder_range[:3]) / torch.tensor(output_shape).type_as(points[0])

        '''
        voxel_ind = torch.arange(output_shape[0] * output_shape[1] * output_shape[2]).to(points[0].device, dtype=torch.long)
        ind0 = voxel_ind // (output_shape[1] * output_shape[2])
        ind1 = voxel_ind % (output_shape[1] * output_shape[2]) // output_shape[2]
        ind2 = voxel_ind % (output_shape[1] * output_shape[2]) % output_shape[2]
        voxel_ind = torch.stack([ind0, ind1, ind2], dim=0).view(-1, *output_shape)
        # add 0.5 ?
        #voxel_position = (voxel_ind.float() + 0.5) * voxel_size[:, None, None, None] + cylinder_range[:3, None, None, None]
        voxel_position = voxel_ind.float() * voxel_size[:, None, None, None] + cylinder_range[:3, None, None, None]
        voxel_position = polar2xyz(voxel_position)
        '''

        pts_feats, coors = [], []
        for pts in points:
            pts_polar = xyz2polar(pts)

            res_coors = torch.floor((pts_polar - cylinder_range[None, :3]) / voxel_size[None, :]).int()
            res_coors[:, 0] = torch.clamp(res_coors[:, 0], min=0, max=output_shape[0]-1)
            res_coors[:, 1] = torch.clamp(res_coors[:, 1], min=0, max=output_shape[1]-1)
            res_coors[:, 2] = torch.clamp(res_coors[:, 2], min=0, max=output_shape[2]-1)

            if return_feat:
                voxel_centers = (res_coors + 0.5) * voxel_size[None, :] + cylinder_range[None, :3]
                voxel_offsets = pts_polar - voxel_centers
                pts_feat = torch.cat((voxel_offsets, pts_polar, pts[:, [0, 1, 3]]), dim=1)
                pts_feats.append(pts_feat)
            else:
                pts_feats.append(pts_polar)
            res_coors = res_coors[:, [2, 1, 0]]
            coors.append(res_coors)
        pts_feats= torch.cat(pts_feats, dim=0)

        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return pts_feats, coors_batch

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""

        point_feats, point_coors = self.voxelize(pts, return_feat=True)

        voxel_feats, voxel_coors = self.voxel_encoder(
            point_feats, point_coors)

        repeating_pillar_coors = voxel_coors.clone()
        repeating_pillar_coors[:, 1] = 0
        pillar_feats, pillar_coors = self.pillar_encoder(voxel_feats, repeating_pillar_coors)

        batch_size = pillar_coors[-1, 0] + 1
        x = self.pts_middle_encoder(pillar_feats, pillar_coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)

        return x, voxel_feats, voxel_coors, point_feats, point_coors

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          pts_semantic_mask=None,
                          voxel_feats=None,
                          voxel_coors=None,
                          point_feats=None,
                          point_coors=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        pred_semantic_cls = self.pts_bbox_head(pts_feats, mode='semantic', voxel_feat=voxel_feats, voxel_coors=voxel_coors)
        loss_inputs = [pts_semantic_mask, point_coors, pred_semantic_cls]
        losses = self.pts_bbox_head.semantic_loss(*loss_inputs)

        return losses

    def simple_test_pts(self,
                        pts_feats,
                        img_metas,
                        rescale=False,
                        voxel_feats=None,
                        voxel_coors=None,
                        point_feats=None,
                        point_coors=None):
        """Test function of point cloud branch."""

        pred_semantic_cls = self.pts_bbox_head(pts_feats, mode='semantic', voxel_feat=voxel_feats, voxel_coors=voxel_coors)
        pred_labels = []

        for _pred_semantic_cls in pred_semantic_cls:
            pred_labels.append(torch.argmax(_pred_semantic_cls, dim=1))

        pred_labels = self.pts_bbox_head.get_point_label_from_voxel(pred_labels, voxel_coors, point_coors)

        '''
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        '''
        return pred_labels

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton.

        The function implementation process is as follows:

            - step 1: map features back for double-flip augmentation.
            - step 2: merge all features and generate boxes.
            - step 3: map boxes back for scale augmentation.
            - step 4: merge results.

        Args:
            feats (list[torch.Tensor]): Feature of point cloud.
            img_metas (list[dict]): Meta information of samples.
            rescale (bool): Whether to rescale bboxes. Default: False.

        Returns:
            dict: Returned bboxes consists of the following keys:

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): Predicted bboxes.
                - scores_3d (torch.Tensor): Scores of predicted boxes.
                - labels_3d (torch.Tensor): Labels of predicted boxes.
        """
        # only support aug_test for one sample
        outs_list = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            # merge augmented outputs before decoding bboxes
            for task_id, out in enumerate(outs):
                for key in out[0].keys():
                    if img_meta[0]['pcd_horizontal_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[2])
                        if key == 'reg':
                            outs[task_id][0][key][:, 1, ...] = 1 - outs[
                                task_id][0][key][:, 1, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                    if img_meta[0]['pcd_vertical_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[3])
                        if key == 'reg':
                            outs[task_id][0][key][:, 0, ...] = 1 - outs[
                                task_id][0][key][:, 0, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]

            outs_list.append(outs)

        preds_dicts = dict()
        scale_img_metas = []

        # concat outputs sharing the same pcd_scale_factor
        for i, (img_meta, outs) in enumerate(zip(img_metas, outs_list)):
            pcd_scale_factor = img_meta[0]['pcd_scale_factor']
            if pcd_scale_factor not in preds_dicts.keys():
                preds_dicts[pcd_scale_factor] = outs
                scale_img_metas.append(img_meta)
            else:
                for task_id, out in enumerate(outs):
                    for key in out[0].keys():
                        preds_dicts[pcd_scale_factor][task_id][0][key] += out[
                            0][key]

        aug_bboxes = []

        for pcd_scale_factor, preds_dict in preds_dicts.items():
            for task_id, pred_dict in enumerate(preds_dict):
                # merge outputs with different flips before decoding bboxes
                for key in pred_dict[0].keys():
                    preds_dict[task_id][0][key] /= len(outs_list) / len(
                        preds_dicts.keys())
            bbox_list = self.pts_bbox_head.get_bboxes(
                preds_dict, img_metas[0], rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        if len(preds_dicts.keys()) > 1:
            # merge outputs with different scales after decoding bboxes
            merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, scale_img_metas,
                                                self.pts_bbox_head.test_cfg)
            return merged_bboxes
        else:
            for key in bbox_list[0].keys():
                bbox_list[0][key] = bbox_list[0][key].to('cpu')
            import pdb
            pdb.set_trace()
            return bbox_list[0]

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            pts_bbox = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=pts_bbox)
        return [bbox_list]
