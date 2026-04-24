import os
import os.path as osp
from tqdm import tqdm

import torch
import numpy as np
import pickle as pkl

from rekognition_online_action_detection.datasets import build_dataset
from rekognition_online_action_detection.evaluation import compute_result
from rekognition_online_action_detection.utils.text_calibration import build_text_calibrator


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

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='BatchInference')
        for batch_idx, data in enumerate(pbar, start=1):
            target = data[-4]
            raw_score = model(*[x.to(device) for x in data[:-4]])   # [B, T, C]

            for bs, (session, query_indices, num_frames) in enumerate(zip(*data[-3:])):
                if session not in pred_scores:
                    pred_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                if session not in gt_targets:
                    gt_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))

                sample_logits = raw_score[bs]   # [T, C]

                if text_calibrator is not None:
                    sample_logits = text_calibrator.calibrate_logits(
                        session=session,
                        vis_logits=sample_logits,
                        query_indices=query_indices,
                    )

                sample_probs = sample_logits.softmax(dim=-1).cpu().numpy()

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
