import os.path as osp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl

from rekognition_online_action_detection.datasets import build_dataset
from rekognition_online_action_detection.evaluation import compute_result
from rekognition_online_action_detection.utils.text_calibration import build_text_calibrator


def _find_classifier_params(model, num_classes):
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and getattr(module, "out_features", None) == num_classes:
            candidates.append((name, module))

    if not candidates:
        raise RuntimeError(f"Cannot find classifier Linear layer with out_features={num_classes}")

    name, module = candidates[-1]
    params = [module.weight]
    if module.bias is not None:
        params.append(module.bias)
    return name, params


def do_perframe_det_batch_inference(cfg, model, device, logger):
    model.eval()
    text_calibrator = build_text_calibrator(cfg, device)

    data_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg, phase='test', tag='BatchInference'),
        batch_size=cfg.DATA_LOADER.BATCH_SIZE * 16,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )

    pred_scores = {}
    gt_targets = {}

    tta_mode = (
        text_calibrator is not None and
        cfg.MODEL.TEXT_CALIBRATION.MODE in ["tta_kl", "tta_kl_topk"]
    )

    if tta_mode:
        cls_name, cls_params = _find_classifier_params(model, cfg.DATA.NUM_CLASSES)
        logger.info(f"[TTA] adapting classifier: {cls_name}")
        base_cls_params = [p.detach().clone() for p in cls_params]
        for p in model.parameters():
            p.requires_grad_(False)
        for p in cls_params:
            p.requires_grad_(True)

    with torch.no_grad() if not tta_mode else torch.enable_grad():
        pbar = tqdm(data_loader, desc='BatchInference')
        for batch_idx, data in enumerate(pbar, start=1):
            target = data[-4]

            if not tta_mode:
                raw_score = model(*[x.to(device) for x in data[:-4]])  # [B, T, C]

                for bs, (session, query_indices, num_frames) in enumerate(zip(*data[-3:])):
                    if session not in pred_scores:
                        pred_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                    if session not in gt_targets:
                        gt_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))

                    sample_logits = raw_score[bs]  # [T, C]

                    if text_calibrator is not None:
                        sample_logits = text_calibrator.calibrate_logits(
                            session=session,
                            vis_logits=sample_logits,
                            query_indices=query_indices,
                        )

                    sample_probs = sample_logits.softmax(dim=-1).detach().cpu().numpy()

                    if query_indices[0] == 0:
                        pred_scores[session][query_indices] = sample_probs
                        gt_targets[session][query_indices] = target[bs]
                    else:
                        pred_scores[session][query_indices[-1]] = sample_probs[-1]
                        gt_targets[session][query_indices[-1]] = target[bs][-1]

            else:
                for bs, (session, query_indices, num_frames) in enumerate(zip(*data[-3:])):
                    if session not in pred_scores:
                        pred_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                    if session not in gt_targets:
                        gt_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))

                    # reset classifier params for each sample window
                    for p, p0 in zip(cls_params, base_cls_params):
                        p.data.copy_(p0.data)

                    sample_inputs = [x[bs:bs+1].to(device) for x in data[:-4]]
                    optimizer = torch.optim.SGD(cls_params, lr=cfg.MODEL.TEXT_CALIBRATION.TTA_LR)

                    for _ in range(cfg.MODEL.TEXT_CALIBRATION.TTA_STEPS):
                        logits = model(*sample_inputs)[0]  # [T, C]

                        q = text_calibrator.get_text_target_distribution(
                            session=session,
                            vis_logits=logits.detach(),
                            query_indices=query_indices,
                            topk_only=(cfg.MODEL.TEXT_CALIBRATION.MODE == "tta_kl_topk"),
                            topk=cfg.MODEL.TEXT_CALIBRATION.TTA_TOPK,
                        )

                        if query_indices[0] == 0:
                            logits_sel = logits
                            q_sel = q
                        else:
                            logits_sel = logits[-1:].contiguous()
                            q_sel = q[-1:].contiguous()

                        loss = F.kl_div(
                            F.log_softmax(logits_sel, dim=-1),
                            q_sel,
                            reduction='batchmean'
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    with torch.no_grad():
                        final_logits = model(*sample_inputs)[0]
                        sample_probs = final_logits.softmax(dim=-1).cpu().numpy()

                    if query_indices[0] == 0:
                        pred_scores[session][query_indices] = sample_probs
                        gt_targets[session][query_indices] = target[bs]
                    else:
                        pred_scores[session][query_indices[-1]] = sample_probs[-1]
                        gt_targets[session][query_indices[-1]] = target[bs][-1]

    pkl.dump({
        'cfg': cfg,
        'perframe_pred_scores': pred_scores,
        'perframe_gt_targets': gt_targets,
    }, open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '.pkl', 'wb'))

    result = compute_result['perframe'](
        cfg,
        np.concatenate(list(gt_targets.values()), axis=0),
        np.concatenate(list(pred_scores.values()), axis=0),
    )
    logger.info('Action detection perframe m{}: {:.5f}'.format(
        cfg.DATA.METRICS, result['mean_AP']
    ))
