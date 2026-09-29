"""Batched MTCNN face detection that keeps the work on the GPU.

This is facenet-pytorch 2.6.0's ``detect_face`` (the detector used to build the thesis
face cache) with one change: stages 2 and 3 crop every candidate box with a single
vectorised gather instead of a Python loop of per-box ``interpolate(mode="area")``
calls. The loop is what kept the CPU busy and the GPU idle.

``interpolate(mode="area")`` is adaptive average pooling. Here each pooling bin is summed
from an integer summed-area table (exact) and divided by the bin height and width, as
PyTorch's kernel does, so the RNet/ONet inputs, and therefore the detections, match the
original implementation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.functional import interpolate
from torchvision.ops.boxes import batched_nms

from facenet_pytorch.models.utils.detect_face import (
    batched_nms_numpy,
    bbreg,
    fixed_batch_process,
    generateBoundingBox,
    rerec,
)

CROP_CHUNK = 2048  # candidate boxes gathered per step (bounds temporary memory)


def _pad(boxes: torch.Tensor, w: int, h: int) -> tuple[torch.Tensor, ...]:
    """facenet ``pad``: truncate to int, clamp the top-left to 1 and bottom-right to the image."""
    b = boxes[:, :4].trunc().int().long()
    x = b[:, 0].clamp(min=1)
    y = b[:, 1].clamp(min=1)
    ex = b[:, 2].clamp(max=w)
    ey = b[:, 3].clamp(max=h)
    return y, ey, x, ex


def _summed_area_table(imgs: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) uint8 -> (B, (H+1)*(W+1), C) int32 exclusive prefix sums (exact for 4K)."""
    b, c, h, w = imgs.shape
    sat = torch.zeros(b, c, h + 1, w + 1, dtype=torch.int32, device=imgs.device)
    sat[:, :, 1:, 1:] = imgs.to(torch.int32).cumsum(2, dtype=torch.int32).cumsum(3, dtype=torch.int32)
    return sat.permute(0, 2, 3, 1).reshape(b, (h + 1) * (w + 1), c)


def _area_crops(sat: torch.Tensor, width1: int, image_inds, y, ey, x, ex, size: int) -> torch.Tensor:
    """Vectorised ``interpolate(imgs[b, :, y-1:ey, x-1:ex], (size, size), mode="area")``."""
    device = sat.device
    top, left = y - 1, x - 1
    height, width = ey - top, ex - left
    i = torch.arange(size, device=device)
    # Adaptive pooling bins: start = floor(i*L/O), end = ceil((i+1)*L/O).
    rs = top[:, None] + (i[None] * height[:, None]) // size
    re = top[:, None] + ((i[None] + 1) * height[:, None] + size - 1) // size
    cs = left[:, None] + (i[None] * width[:, None]) // size
    ce = left[:, None] + ((i[None] + 1) * width[:, None] + size - 1) // size
    b = image_inds.long()[:, None, None]

    def corner(r, c):
        return sat[b, r[:, :, None] * width1 + c[:, None, :]]  # (N, O, O, C) int32

    # Subtract in exact integers first: the table values exceed float32's exact range.
    sums = (corner(re, ce) - corner(rs, ce) - corner(re, cs) + corner(rs, cs)).float()
    kh = (re - rs).float()[:, :, None, None]
    kw = (ce - cs).float()[:, None, :, None]
    return (sums / kh / kw).permute(0, 3, 1, 2).contiguous()  # (N, C, O, O)


def _crops(imgs_u8: torch.Tensor, sat_cache: dict, image_inds, y, ey, x, ex, size: int) -> torch.Tensor:
    if "sat" not in sat_cache:
        sat_cache["sat"] = _summed_area_table(imgs_u8)
    width1 = imgs_u8.shape[3] + 1
    out = [
        _area_crops(sat_cache["sat"], width1, image_inds[s : s + CROP_CHUNK], y[s : s + CROP_CHUNK],
                    ey[s : s + CROP_CHUNK], x[s : s + CROP_CHUNK], ex[s : s + CROP_CHUNK], size)
        for s in range(0, len(y), CROP_CHUNK)
    ]
    return torch.cat(out, dim=0)


@torch.no_grad()
def detect_faces(imgs_u8: torch.Tensor, minsize: int, pnet, rnet, onet, threshold, factor):
    """Detect faces in a batch of equally sized RGB frames.

    imgs_u8: (B, 3, H, W) uint8 tensor on the detector's device.
    Returns per-frame lists: boxes (n, 4), probs (n,), points (n, 5, 2) as float32 numpy,
    ordered exactly like ``MTCNN.detect`` (keep_all=True, select_largest=False).
    """
    device = imgs_u8.device
    imgs = imgs_u8.float()
    batch_size = len(imgs)
    h, w = imgs.shape[2:4]
    m = 12.0 / minsize
    minl = min(h, w) * m

    scale_i = m
    scales = []
    while minl >= 12:
        scales.append(scale_i)
        scale_i = scale_i * factor
        minl = minl * factor

    # First stage (already batched in facenet-pytorch).
    boxes, image_inds, scale_picks = [], [], []
    offset = 0
    for scale in scales:
        im_data = interpolate(imgs, size=(int(h * scale + 1), int(w * scale + 1)), mode="area")
        im_data = (im_data - 127.5) * 0.0078125
        reg, probs = pnet(im_data)
        boxes_scale, image_inds_scale = generateBoundingBox(reg, probs[:, 1], scale, threshold[0])
        boxes.append(boxes_scale)
        image_inds.append(image_inds_scale)
        pick = batched_nms(boxes_scale[:, :4], boxes_scale[:, 4], image_inds_scale, 0.5)
        scale_picks.append(pick + offset)
        offset += boxes_scale.shape[0]
    del imgs

    boxes = torch.cat(boxes, dim=0)
    image_inds = torch.cat(image_inds, dim=0)
    scale_picks = torch.cat(scale_picks, dim=0)
    boxes, image_inds = boxes[scale_picks], image_inds[scale_picks]
    pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    boxes, image_inds = boxes[pick], image_inds[pick]

    regw = boxes[:, 2] - boxes[:, 0]
    regh = boxes[:, 3] - boxes[:, 1]
    qq1 = boxes[:, 0] + boxes[:, 5] * regw
    qq2 = boxes[:, 1] + boxes[:, 6] * regh
    qq3 = boxes[:, 2] + boxes[:, 7] * regw
    qq4 = boxes[:, 3] + boxes[:, 8] * regh
    boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
    boxes = rerec(boxes)

    sat_cache: dict = {}

    # Second stage: one gather for all candidate boxes.
    if len(boxes) > 0:
        y, ey, x, ex = _pad(boxes, w, h)
        ok = (ey > y - 1) & (ex > x - 1)  # always true in practice; keeps arrays aligned
        boxes, image_inds, y, ey, x, ex = boxes[ok], image_inds[ok], y[ok], ey[ok], x[ok], ex[ok]
    if len(boxes) > 0:
        im_data = _crops(imgs_u8, sat_cache, image_inds, y, ey, x, ex, 24)
        im_data = (im_data - 127.5) * 0.0078125
        out = fixed_batch_process(im_data, rnet)
        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        score = out1[1, :]
        ipass = score > threshold[1]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
        boxes = bbreg(boxes, mv)
        boxes = rerec(boxes)

    # Third stage.
    points = torch.zeros(0, 5, 2, device=device)
    if len(boxes) > 0:
        y, ey, x, ex = _pad(boxes, w, h)
        ok = (ey > y - 1) & (ex > x - 1)
        boxes, image_inds, y, ey, x, ex = boxes[ok], image_inds[ok], y[ok], ey[ok], x[ok], ex[ok]
    if len(boxes) > 0:
        im_data = _crops(imgs_u8, sat_cache, image_inds, y, ey, x, ex, 48)
        im_data = (im_data - 127.5) * 0.0078125
        out = fixed_batch_process(im_data, onet)
        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        out2 = out[2].permute(1, 0)
        score = out2[1, :]
        points = out1
        ipass = score > threshold[2]
        points = points[:, ipass]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        w_i = boxes[:, 2] - boxes[:, 0] + 1
        h_i = boxes[:, 3] - boxes[:, 1] + 1
        points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
        points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
        points = torch.stack((points_x, points_y)).permute(2, 1, 0)
        boxes = bbreg(boxes, mv)
        # Final "Min" NMS runs on the few surviving boxes, as in facenet-pytorch.
        pick = batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds, 0.7, "Min")
        boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]
    sat_cache.clear()

    boxes = boxes.cpu().numpy()
    points = points.cpu().numpy()
    image_inds = image_inds.cpu().numpy()
    results = []
    for b_i in range(batch_size):
        sel = np.where(image_inds == b_i)[0]
        results.append((boxes[sel, :4].copy(), boxes[sel, 4].copy(), points[sel].copy()))
    return results
