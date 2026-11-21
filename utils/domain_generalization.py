"""Utilities for strict source-only cross-subject domain generalization."""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler


class SubjectBalancedBatchSampler(Sampler[list[int]]):
    """Yield batches with an equal number of trials from every source subject."""

    def __init__(
        self,
        subject_ids: np.ndarray | torch.Tensor,
        batch_size: int,
        seed: int = 0,
    ) -> None:
        subject_ids = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
        self.subject_ids = subject_ids
        self.subjects = np.unique(subject_ids)
        if self.subjects.size == 0:
            raise ValueError("subject_ids cannot be empty")
        if batch_size < self.subjects.size:
            raise ValueError("batch_size must be at least the number of subjects")
        self.samples_per_subject = batch_size // self.subjects.size
        self.effective_batch_size = self.samples_per_subject * self.subjects.size
        self.indices = {
            int(subject): np.flatnonzero(subject_ids == subject)
            for subject in self.subjects
        }
        largest_subject = max(len(indices) for indices in self.indices.values())
        self.n_batches = math.ceil(largest_subject / self.samples_per_subject)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.n_batches

    @staticmethod
    def _draw_epoch_indices(
        indices: np.ndarray,
        required: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        chunks = []
        remaining = required
        while remaining > 0:
            shuffled = rng.permutation(indices)
            take = min(remaining, shuffled.size)
            chunks.append(shuffled[:take])
            remaining -= take
        return np.concatenate(chunks)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        required = self.n_batches * self.samples_per_subject
        subject_batches = {}
        for subject, indices in self.indices.items():
            drawn = self._draw_epoch_indices(indices, required, rng)
            subject_batches[subject] = drawn.reshape(
                self.n_batches, self.samples_per_subject
            )

        for batch_index in range(self.n_batches):
            batch = np.concatenate(
                [subject_batches[int(subject)][batch_index] for subject in self.subjects]
            )
            rng.shuffle(batch)
            yield batch.tolist()


def classwise_subject_augmentation(
    data: torch.Tensor,
    labels: torch.Tensor,
    subject_ids: torch.Tensor,
    n_segments: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Segment/reconstruct trials only within the same subject and class."""

    if data.ndim != 4:
        raise ValueError("data must be shaped (B,1,C,T)")
    if labels.numel() != data.shape[0] or subject_ids.numel() != data.shape[0]:
        raise ValueError("data, labels, and subject_ids must have the same batch size")

    _, _, channels, n_times = data.shape
    segment_length = n_times // n_segments
    if segment_length == 0:
        return data, labels, subject_ids

    augmented_data = []
    augmented_labels = []
    augmented_subjects = []
    for subject in torch.unique(subject_ids):
        subject_mask = subject_ids == subject
        for label in torch.unique(labels[subject_mask]):
            group_indices = torch.nonzero(
                subject_mask & (labels == label), as_tuple=True
            )[0]
            group = data.index_select(0, group_indices)
            n_augmented = group.shape[0]
            reconstructed = torch.zeros(
                (n_augmented, 1, channels, n_times),
                device=data.device,
                dtype=data.dtype,
            )
            random_indices = torch.randint(
                group.shape[0],
                (n_augmented, n_segments),
                device=data.device,
            )
            for segment in range(n_segments):
                start = segment * segment_length
                end = n_times if segment == n_segments - 1 else start + segment_length
                selected = group.index_select(0, random_indices[:, segment])
                reconstructed[..., start:end] = selected[..., start:end]

            augmented_data.append(reconstructed)
            augmented_labels.append(
                torch.full(
                    (n_augmented,), label, device=labels.device, dtype=labels.dtype
                )
            )
            augmented_subjects.append(
                torch.full(
                    (n_augmented,),
                    subject,
                    device=subject_ids.device,
                    dtype=subject_ids.dtype,
                )
            )

    if not augmented_data:
        return data, labels, subject_ids

    augmented_data_tensor = torch.cat(augmented_data, dim=0)
    augmented_labels_tensor = torch.cat(augmented_labels, dim=0)
    augmented_subjects_tensor = torch.cat(augmented_subjects, dim=0)
    permutation = torch.randperm(augmented_data_tensor.shape[0], device=data.device)
    return (
        torch.cat([data, augmented_data_tensor[permutation]], dim=0),
        torch.cat([labels, augmented_labels_tensor[permutation]], dim=0),
        torch.cat([subject_ids, augmented_subjects_tensor[permutation]], dim=0),
    )


@dataclass
class WeightSegment:
    state: OrderedDict[str, torch.Tensor]
    n_averaged: int
    start_step: int
    end_step: int
    epoch: int
    val_loss: float


class DenseWeightAverager:
    """Average every optimization step in one validation segment."""

    def __init__(self) -> None:
        self.state: OrderedDict[str, torch.Tensor] | None = None
        self.n_averaged = 0
        self.start_step = -1
        self.end_step = -1

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        model_state = model.state_dict()
        if self.state is None:
            self.state = OrderedDict(
                (name, value.detach().clone()) for name, value in model_state.items()
            )
            self.start_step = int(step)
            self.n_averaged = 1
        else:
            count = self.n_averaged + 1
            for name, value in model_state.items():
                if torch.is_floating_point(self.state[name]):
                    self.state[name].add_(
                        (value.detach() - self.state[name]) / float(count)
                    )
                else:
                    self.state[name].copy_(value.detach())
            self.n_averaged = count
        self.end_step = int(step)

    def freeze(self, epoch: int, val_loss: float) -> WeightSegment:
        if self.state is None:
            raise RuntimeError("Cannot freeze an empty weight average")
        return WeightSegment(
            state=OrderedDict(
                (name, value.detach().cpu().clone())
                for name, value in self.state.items()
            ),
            n_averaged=self.n_averaged,
            start_step=self.start_step,
            end_step=self.end_step,
            epoch=int(epoch),
            val_loss=float(val_loss),
        )


class WeightedModelAverage:
    """Combine segment averages while respecting their optimization-step counts."""

    def __init__(self) -> None:
        self.state: OrderedDict[str, torch.Tensor] | None = None
        self.n_averaged = 0
        self.start_step = -1
        self.end_step = -1

    @torch.no_grad()
    def add(self, segment: WeightSegment) -> None:
        if self.state is None:
            self.state = OrderedDict(
                (name, value.clone()) for name, value in segment.state.items()
            )
            self.n_averaged = segment.n_averaged
            self.start_step = segment.start_step
            self.end_step = segment.end_step
            return

        old_count = self.n_averaged
        new_count = old_count + segment.n_averaged
        fraction = segment.n_averaged / float(new_count)
        for name, value in segment.state.items():
            if torch.is_floating_point(self.state[name]):
                self.state[name].add_((value - self.state[name]) * fraction)
            else:
                self.state[name].copy_(value)
        self.n_averaged = new_count
        self.end_step = segment.end_step


class LossValleySWAD:
    """Dense, validation-loss-valley weight averaging for epoch checkpoints.

    Each checkpoint represents the dense average of all optimizer steps since
    the preceding validation.  The convergence and tolerance defaults follow
    the official SWAD LossValley configuration (3, 6, 0.3).
    """

    def __init__(
        self,
        n_converge: int = 3,
        n_tolerance: int = 6,
        tolerance_ratio: float = 0.3,
    ) -> None:
        if n_converge < 2 or n_tolerance < 1:
            raise ValueError("Invalid SWAD window size")
        self.n_converge = int(n_converge)
        self.n_tolerance = int(n_tolerance)
        self.tolerance_ratio = float(tolerance_ratio)
        self.convergence_queue: deque[WeightSegment] = deque(maxlen=n_converge)
        self.history: deque[WeightSegment] = deque(maxlen=n_tolerance)
        self.pending: deque[WeightSegment] = deque()
        self.average = WeightedModelAverage()
        self.converged = False
        self.dead_valley = False
        self.converge_epoch: int | None = None
        self.threshold: float | None = None

    def update(self, segment: WeightSegment) -> None:
        if self.dead_valley:
            return
        self.history.append(segment)
        self.convergence_queue.append(segment)

        if not self.converged:
            if len(self.convergence_queue) < self.n_converge:
                return
            losses = [item.val_loss for item in self.convergence_queue]
            if int(np.argmin(losses)) != 0:
                return

            self.converged = True
            first_valley_segment = self.convergence_queue[0]
            self.converge_epoch = first_valley_segment.epoch
            self.threshold = float(np.mean(losses) * (1.0 + self.tolerance_ratio))

            recent = list(self.history)
            converge_position = next(
                index
                for index, item in enumerate(recent)
                if item.epoch == first_valley_segment.epoch
            )
            start_position = 0
            for index in range(converge_position):
                if recent[index].val_loss > self.threshold:
                    start_position = index + 1
            self.pending.extend(recent[start_position:])
        else:
            self.pending.append(segment)

        self._commit_safe_prefix()

    def _commit_safe_prefix(self) -> None:
        if not self.converged or self.threshold is None:
            return
        while len(self.pending) >= self.n_tolerance:
            window = list(self.pending)[: self.n_tolerance]
            if min(item.val_loss for item in window) > self.threshold:
                self.dead_valley = True
                self.pending.clear()
                return
            self.average.add(self.pending.popleft())

    def finalize(self) -> OrderedDict[str, torch.Tensor] | None:
        if not self.converged or self.threshold is None:
            return None
        while self.pending:
            if min(item.val_loss for item in self.pending) > self.threshold:
                break
            self.average.add(self.pending.popleft())
        return self.average.state

    @property
    def averaged_range(self) -> tuple[int, int, int] | None:
        if self.average.state is None:
            return None
        return (
            self.average.start_step,
            self.average.end_step,
            self.average.n_averaged,
        )


@torch.no_grad()
def refresh_batch_norm(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """Recompute BatchNorm buffers after weight averaging, without MixStyle."""

    batch_norms = [
        module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    if not batch_norms:
        return

    momenta = {module: module.momentum for module in batch_norms}
    for module in batch_norms:
        module.reset_running_stats()
        module.momentum = None

    mixstyle_probability = None
    if hasattr(model, "mixstyle"):
        mixstyle_probability = model.mixstyle.probability
        model.mixstyle.probability = 0.0

    was_training = model.training
    model.train()
    for batch in loader:
        inputs = batch[0].to(device, non_blocking=True)
        model(inputs)

    for module, momentum in momenta.items():
        module.momentum = momentum
    if mixstyle_probability is not None:
        model.mixstyle.probability = mixstyle_probability
    model.train(was_training)
