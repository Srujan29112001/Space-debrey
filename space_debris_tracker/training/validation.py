"""
Validation Metrics for Space Debris Tracking Models
Comprehensive evaluation for detection, trajectory prediction, and risk assessment
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class DetectionMetrics:
    """Detection evaluation metrics"""
    precision: float
    recall: float
    f1_score: float
    map_50: float  # mAP at IoU=0.5
    map_75: float  # mAP at IoU=0.75
    map_50_95: float  # mAP averaged over IoU=0.5:0.95
    fps: float  # Frames per second


@dataclass
class TrajectoryMetrics:
    """Trajectory prediction metrics"""
    position_rmse: float  # km
    velocity_rmse: float  # km/s
    position_mae: float  # km
    velocity_mae: float  # km/s
    along_track_error: float  # km
    cross_track_error: float  # km
    radial_error: float  # km
    horizon_errors: Dict[int, float]  # Error at different time horizons


@dataclass
class ConjunctionMetrics:
    """Conjunction/collision risk metrics"""
    roc_auc: float
    pr_auc: float  # Precision-Recall AUC
    brier_score: float
    calibration_error: float
    true_positive_rate: float
    false_positive_rate: float
    detection_rate_at_1e4: float  # Detection rate at FAR=1e-4


class DetectionEvaluator:
    """
    Evaluator for object detection models (YOLOv7)
    """

    def __init__(self, num_classes: int, iou_thresholds: List[float] = None):
        """
        Initialize detection evaluator

        Args:
            num_classes: Number of object classes
            iou_thresholds: IoU thresholds for mAP calculation
        """
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5, 0.75] + list(np.arange(0.5, 1.0, 0.05))

    def compute_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute IoU between two sets of boxes

        Args:
            boxes1: [N, 4] (x1, y1, x2, y2)
            boxes2: [M, 4] (x1, y1, x2, y2)

        Returns:
            IoU matrix [N, M]
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
        inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
        inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
        inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        union_area = area1[:, None] + area2[None, :] - inter_area

        iou = inter_area / (union_area + 1e-6)

        return iou

    def compute_ap(
        self,
        predictions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        ground_truths: List[Tuple[torch.Tensor, torch.Tensor]],
        iou_threshold: float = 0.5
    ) -> float:
        """
        Compute Average Precision

        Args:
            predictions: List of (boxes, scores, labels) for each image
            ground_truths: List of (boxes, labels) for each image
            iou_threshold: IoU threshold

        Returns:
            Average Precision
        """
        all_scores = []
        all_matches = []

        for pred, gt in zip(predictions, ground_truths):
            pred_boxes, pred_scores, pred_labels = pred
            gt_boxes, gt_labels = gt

            if len(pred_boxes) == 0:
                continue

            if len(gt_boxes) == 0:
                # All predictions are false positives
                all_scores.extend(pred_scores.cpu().numpy())
                all_matches.extend([0] * len(pred_scores))
                continue

            # Compute IoU
            ious = self.compute_iou(pred_boxes, gt_boxes)

            # Match predictions to ground truths
            matched_gt = set()
            for i, (score, label) in enumerate(zip(pred_scores, pred_labels)):
                all_scores.append(score.item())

                # Find best matching ground truth
                class_mask = (gt_labels == label)
                if not class_mask.any():
                    all_matches.append(0)
                    continue

                class_ious = ious[i, class_mask]
                best_iou, best_idx = class_ious.max(0)

                # Map back to original gt index
                gt_indices = torch.where(class_mask)[0]
                best_gt_idx = gt_indices[best_idx].item()

                if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                    all_matches.append(1)
                    matched_gt.add(best_gt_idx)
                else:
                    all_matches.append(0)

        if len(all_scores) == 0:
            return 0.0

        # Sort by score
        indices = np.argsort(all_scores)[::-1]
        matches = np.array(all_matches)[indices]
        scores = np.array(all_scores)[indices]

        # Compute precision and recall
        tp = np.cumsum(matches)
        fp = np.cumsum(1 - matches)
        n_gt = sum(len(gt[0]) for gt in ground_truths)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (n_gt + 1e-6)

        # Compute AP (area under precision-recall curve)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recall >= t
            if mask.any():
                ap += precision[mask].max() / 11

        return ap

    def evaluate(
        self,
        predictions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        ground_truths: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> DetectionMetrics:
        """
        Evaluate detection model

        Args:
            predictions: List of (boxes, scores, labels)
            ground_truths: List of (boxes, labels)

        Returns:
            Detection metrics
        """
        # Compute mAP at different IoU thresholds
        map_50 = self.compute_ap(predictions, ground_truths, iou_threshold=0.5)
        map_75 = self.compute_ap(predictions, ground_truths, iou_threshold=0.75)

        # Compute mAP@[0.5:0.95]
        aps = []
        for iou_thresh in np.arange(0.5, 1.0, 0.05):
            ap = self.compute_ap(predictions, ground_truths, iou_threshold=iou_thresh)
            aps.append(ap)
        map_50_95 = np.mean(aps)

        # Compute precision and recall at IoU=0.5
        all_matches = []
        all_pred_count = 0
        all_gt_count = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_boxes, pred_scores, pred_labels = pred
            gt_boxes, gt_labels = gt

            all_pred_count += len(pred_boxes)
            all_gt_count += len(gt_boxes)

            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue

            ious = self.compute_iou(pred_boxes, gt_boxes)
            matched_gt = set()

            for i in range(len(pred_boxes)):
                best_iou, best_idx = ious[i].max(0)
                if best_iou >= 0.5 and best_idx.item() not in matched_gt:
                    all_matches.append(1)
                    matched_gt.add(best_idx.item())
                else:
                    all_matches.append(0)

        tp = sum(all_matches)
        precision = tp / (all_pred_count + 1e-6)
        recall = tp / (all_gt_count + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            map_50=map_50,
            map_75=map_75,
            map_50_95=map_50_95,
            fps=0.0  # Should be measured separately
        )


class TrajectoryEvaluator:
    """
    Evaluator for trajectory prediction models (PINN, Transformer, Mamba)
    """

    def __init__(self):
        """Initialize trajectory evaluator"""
        pass

    def compute_horizon_errors(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        horizons: List[int]
    ) -> Dict[int, float]:
        """
        Compute position errors at different time horizons

        Args:
            predictions: [B, T, 6]
            targets: [B, T, 6]
            horizons: List of time steps to evaluate

        Returns:
            Dictionary mapping horizon to RMSE
        """
        errors = {}

        for h in horizons:
            if h <= predictions.shape[1]:
                pred_pos = predictions[:, h-1, :3]
                targ_pos = targets[:, h-1, :3]
                rmse = torch.sqrt(torch.mean((pred_pos - targ_pos) ** 2)).item()
                errors[h] = rmse

        return errors

    def compute_rtc_errors(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Compute Radial, Along-Track, Cross-Track errors

        Args:
            predictions: [B, T, 6] (positions and velocities)
            targets: [B, T, 6]

        Returns:
            (radial_error, along_track_error, cross_track_error)
        """
        pred_pos = predictions[:, :, :3]  # [B, T, 3]
        targ_pos = targets[:, :, :3]

        pred_vel = predictions[:, :, 3:]
        targ_vel = targets[:, :, 3:]

        # Position errors
        pos_error = pred_pos - targ_pos  # [B, T, 3]

        # Radial direction (from Earth center)
        r_hat = targ_pos / (torch.norm(targ_pos, dim=2, keepdim=True) + 1e-8)

        # Cross-track direction (perpendicular to orbital plane)
        h = torch.cross(targ_pos, targ_vel, dim=2)  # Angular momentum
        c_hat = h / (torch.norm(h, dim=2, keepdim=True) + 1e-8)

        # Along-track direction
        a_hat = torch.cross(c_hat, r_hat, dim=2)
        a_hat = a_hat / (torch.norm(a_hat, dim=2, keepdim=True) + 1e-8)

        # Project errors onto RTC frame
        radial_error = torch.sum(pos_error * r_hat, dim=2)
        along_track_error = torch.sum(pos_error * a_hat, dim=2)
        cross_track_error = torch.sum(pos_error * c_hat, dim=2)

        # RMS errors
        radial_rms = torch.sqrt(torch.mean(radial_error ** 2)).item()
        along_track_rms = torch.sqrt(torch.mean(along_track_error ** 2)).item()
        cross_track_rms = torch.sqrt(torch.mean(cross_track_error ** 2)).item()

        return radial_rms, along_track_rms, cross_track_rms

    def evaluate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        horizons: Optional[List[int]] = None
    ) -> TrajectoryMetrics:
        """
        Evaluate trajectory prediction

        Args:
            predictions: [B, T, 6]
            targets: [B, T, 6]
            horizons: Time horizons to evaluate

        Returns:
            Trajectory metrics
        """
        if horizons is None:
            horizons = [1, 10, 50, 100]

        # Position and velocity RMSE/MAE
        position_rmse = torch.sqrt(torch.mean(
            (predictions[:, :, :3] - targets[:, :, :3]) ** 2
        )).item()

        velocity_rmse = torch.sqrt(torch.mean(
            (predictions[:, :, 3:] - targets[:, :, 3:]) ** 2
        )).item()

        position_mae = torch.mean(torch.abs(
            predictions[:, :, :3] - targets[:, :, :3]
        )).item()

        velocity_mae = torch.mean(torch.abs(
            predictions[:, :, 3:] - targets[:, :, 3:]
        )).item()

        # RTC errors
        radial, along_track, cross_track = self.compute_rtc_errors(
            predictions, targets
        )

        # Horizon errors
        horizon_errors = self.compute_horizon_errors(
            predictions, targets, horizons
        )

        return TrajectoryMetrics(
            position_rmse=position_rmse,
            velocity_rmse=velocity_rmse,
            position_mae=position_mae,
            velocity_mae=velocity_mae,
            along_track_error=along_track,
            cross_track_error=cross_track,
            radial_error=radial,
            horizon_errors=horizon_errors
        )


class ConjunctionEvaluator:
    """
    Evaluator for conjunction/collision risk prediction
    """

    def __init__(self):
        """Initialize conjunction evaluator"""
        pass

    def compute_calibration_error(
        self,
        predicted_probs: np.ndarray,
        true_labels: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE)

        Args:
            predicted_probs: Predicted probabilities
            true_labels: True binary labels
            n_bins: Number of bins

        Returns:
            Calibration error
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predicted_probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_conf = predicted_probs[mask].mean()
                bin_acc = true_labels[mask].mean()
                bin_weight = mask.sum() / len(predicted_probs)
                ece += bin_weight * abs(bin_conf - bin_acc)

        return ece

    def compute_brier_score(
        self,
        predicted_probs: np.ndarray,
        true_labels: np.ndarray
    ) -> float:
        """
        Compute Brier score

        Args:
            predicted_probs: Predicted probabilities
            true_labels: True binary labels

        Returns:
            Brier score
        """
        return np.mean((predicted_probs - true_labels) ** 2)

    def evaluate(
        self,
        predicted_probs: np.ndarray,
        true_labels: np.ndarray,
        threshold: float = 0.5
    ) -> ConjunctionMetrics:
        """
        Evaluate conjunction prediction

        Args:
            predicted_probs: Predicted collision probabilities
            true_labels: True binary labels (1=collision, 0=safe)
            threshold: Classification threshold

        Returns:
            Conjunction metrics
        """
        # ROC-AUC
        roc_auc = roc_auc_score(true_labels, predicted_probs)

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)
        pr_auc = auc(recall, precision)

        # Brier score
        brier_score = self.compute_brier_score(predicted_probs, true_labels)

        # Calibration error
        calibration_error = self.compute_calibration_error(
            predicted_probs, true_labels
        )

        # Classification metrics at threshold
        predictions = (predicted_probs >= threshold).astype(int)
        tp = np.sum((predictions == 1) & (true_labels == 1))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        tn = np.sum((predictions == 0) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))

        tpr = tp / (tp + fn + 1e-6)
        fpr = fp / (fp + tn + 1e-6)

        # Detection rate at specific FAR
        sorted_indices = np.argsort(-predicted_probs)
        sorted_labels = true_labels[sorted_indices]

        n_false_alarms = np.sum(true_labels == 0)
        far_threshold_idx = int(n_false_alarms * 1e-4)

        if far_threshold_idx < len(sorted_labels):
            detection_rate = np.mean(sorted_labels[:far_threshold_idx])
        else:
            detection_rate = 0.0

        return ConjunctionMetrics(
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            brier_score=brier_score,
            calibration_error=calibration_error,
            true_positive_rate=tpr,
            false_positive_rate=fpr,
            detection_rate_at_1e4=detection_rate
        )


def plot_calibration_curve(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[Path] = None
):
    """
    Plot calibration curve

    Args:
        predicted_probs: Predicted probabilities
        true_labels: True binary labels
        n_bins: Number of bins
        save_path: Path to save plot
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predicted_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_means = []
    bin_accs = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(predicted_probs[mask].mean())
            bin_accs.append(true_labels[mask].mean())

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(bin_means, bin_accs, 'o-', label='Model')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    print("Testing validation metrics...")

    # Test detection evaluator
    print("\n1. Detection Evaluator")
    det_eval = DetectionEvaluator(num_classes=4)

    # Dummy predictions and ground truths
    predictions = [
        (
            torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]]),
            torch.tensor([0.9, 0.8]),
            torch.tensor([0, 1])
        )
    ]
    ground_truths = [
        (
            torch.tensor([[12, 12, 48, 48], [58, 58, 98, 98]]),
            torch.tensor([0, 1])
        )
    ]

    det_metrics = det_eval.evaluate(predictions, ground_truths)
    print(f"  Precision: {det_metrics.precision:.3f}")
    print(f"  Recall: {det_metrics.recall:.3f}")
    print(f"  mAP@0.5: {det_metrics.map_50:.3f}")

    # Test trajectory evaluator
    print("\n2. Trajectory Evaluator")
    traj_eval = TrajectoryEvaluator()

    predictions = torch.randn(10, 50, 6) * 7000
    targets = predictions + torch.randn(10, 50, 6) * 10

    traj_metrics = traj_eval.evaluate(predictions, targets)
    print(f"  Position RMSE: {traj_metrics.position_rmse:.3f} km")
    print(f"  Velocity RMSE: {traj_metrics.velocity_rmse:.6f} km/s")
    print(f"  Radial error: {traj_metrics.radial_error:.3f} km")

    # Test conjunction evaluator
    print("\n3. Conjunction Evaluator")
    conj_eval = ConjunctionEvaluator()

    predicted_probs = np.random.rand(1000)
    true_labels = (np.random.rand(1000) > 0.8).astype(int)

    conj_metrics = conj_eval.evaluate(predicted_probs, true_labels)
    print(f"  ROC-AUC: {conj_metrics.roc_auc:.3f}")
    print(f"  PR-AUC: {conj_metrics.pr_auc:.3f}")
    print(f"  Brier score: {conj_metrics.brier_score:.3f}")
    print(f"  Calibration error: {conj_metrics.calibration_error:.3f}")

    print("\nAll tests completed!")
