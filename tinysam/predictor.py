# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from tinysam.modeling import Sam
from torch.autograd import profiler

from typing import Optional, Tuple

from .utils.transforms import ResizeLongestSide

from .tome import bipartite_soft_matching_random2d, merge_source, make_visualization, bipartite_soft_matching

from torchvision.transforms import ToPILImage
from torchvision import transforms

class SamPredictor:
    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        with profiler.record_function("preprocess"):   # 忽略不计
          input_image = self.model.preprocess(transformed_image)
        with profiler.record_function("image_encoder"):   # 忽略不计
          self.features = self.model.image_encoder(input_image)
          
        # from .tome import compute_cosine_similarity, plot_similarity_matrix, visualize_cosine_similarity
        # # 计算余弦相似度矩阵
        # cosine_sim_matrix = compute_cosine_similarity(self.features.permute(0, 2, 3, 1).view(1, 4096, 256))  # 取batch中的第一个元素
        # visualize_cosine_similarity(cosine_sim_matrix[0].cpu(), H=30, W=15)
        # exit()
        # # 绘制余弦相似度矩阵的颜色图
        # plot_similarity_matrix(cosine_sim_matrix.cpu())
        
        #===================  Token Merging - Start  =====================#
        tome_feature = self.features.permute(0, 2, 3, 1).view(1, 4096, 256)
        org_feature = tome_feature.clone()
        org_norm = torch.norm(org_feature, p=2)
        
        def adjust_blocks_to_merge(initial_blocks, final_blocks, total_iterations, current_iteration):
            """
            Adjust the number of blocks to merge based on the current iteration.
            initial_blocks: Number of blocks to merge at the start.
            final_blocks: Minimum number of blocks to merge at the end.
            total_iterations: Total number of iterations.
            current_iteration: Current iteration number.
            """
            # Linearly decrease the number of blocks to merge.
            blocks_to_merge = initial_blocks - (initial_blocks - final_blocks) * (current_iteration / total_iterations)
            return max(blocks_to_merge, final_blocks)

        def feature_difference(org_feature, tome_feature, sources):
            """
            Calculate the difference between the original feature map and the current feature map.
            org_feature: The original feature map, shape [1, 4096, 256].
            tome_feature: The current feature map, shape [1, N, 256] where N <= 4096.
            sources: The sources mapping, shape [1, N, 4096].
            """
            # Reconstruct the current feature to match the original feature's dimension
            reconstructed_feature = reconstruct_feature(tome_feature, sources)
            
            return torch.norm(reconstructed_feature - org_feature, p=2)
          
        def reconstruct_feature(tome_feature, sources):
            """
            Reconstruct the current feature map to match the original dimension.
            tome_feature: The current feature map, shape [1, 4016, 256].
            sources: The sources mapping, shape [1, 4016, 4096].
            """
            # Expand tome_feature to match the last dimension of sources
            expanded_tome_feature = tome_feature.unsqueeze(3)  # Shape becomes [1, 4016, 256, 1]

            # Multiply the expanded tome_feature with sources
            # sources is expanded to match the third dimension of expanded_tome_feature
            weighted_features = expanded_tome_feature * sources.unsqueeze(2)  # Shape becomes [1, 4016, 256, 4096]

            # Sum over the second dimension to combine contributions from each new feature to the original features
            reconstructed_feature = torch.sum(weighted_features, dim=1)  # Shape becomes [1, 256, 4096]

            # Transpose to match the shape of org_feature
            reconstructed_feature = reconstructed_feature.transpose(1, 2)  # Shape becomes [1, 4096, 256]

            return reconstructed_feature

        # Example usage in your main loop
        initial_blocks_to_merge = 80  # Start with a higher number
        final_blocks_to_merge = 10    # End with a lower number
        total_iterations = 100        # Total iterations
        stop_threshold = 0.8  # Set a threshold for stopping

        for i in range(total_iterations):
            blocks_to_merge = adjust_blocks_to_merge(initial_blocks_to_merge, final_blocks_to_merge, total_iterations, i)
            merge, _ = bipartite_soft_matching(tome_feature, int(blocks_to_merge), False, False)
            sources = merge_source(merge, tome_feature, sources if i else None)
            tome_feature = merge(tome_feature)
            diff = feature_difference(org_feature, tome_feature, sources) / org_norm
            if diff > stop_threshold:
                break
              
            print(f"Iter-{i}: tokens={tome_feature.shape[1]}, diff={diff * 100:.1f}%")
            
        from torch.nn import functional as F
        import pdb;pdb.set_trace()
        
        h, w = transformed_image.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        img_vis = F.pad(transformed_image, (0, padw, 0, padh))[0,].permute(1, 2, 0).cpu()
        make_visualization(img_vis, sources, patch_size=16, class_token=False)
        exit()
        #===================  Token Merging - End =====================#
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None
            
        with profiler.record_function("prompt_encoder"):   # 
            # Embed prompts
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=mask_input,
            )

        with profiler.record_function("mask_decoder"):   # 
            # Predict masks
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=self.features,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

        with profiler.record_function("postprocess_masks"):   # 
            # Upscale the masks to the original image resolution
            masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
