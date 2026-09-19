"""Torch detector reductions over borrowed bounded-read sources."""

import math

import torch


class BoundedDetectorCompute:
    """Reduce native regions without materializing a complete measurement."""

    def __init__(self, data):
        self.data = data
        self.scan_shape = tuple(data.shape[:2])
        self.det_shape = tuple(data.shape[2:])
        self.n_frames = math.prod(self.scan_shape)
        self.device = torch.device(data.device)
        self.capabilities = ()
        self._native = None
        from quantem.gpu.io.models import FourDSTEMData
        owner = getattr(data, '_detector_source', None)
        if isinstance(owner, FourDSTEMData) and owner.representation == 'encoded':
            from quantem.gpu.detector import prepare

            self._native = prepare(owner)

    def _blocks(self):
        for row in range(self.scan_shape[0]):
            for col in range(0, self.scan_shape[1], 32):
                stop = min(col + 32, self.scan_shape[1])
                yield row, col, stop, self.data.read(
                    scan_region=(row, row + 1, col, stop)
                ).float()

    def frame(self, index):
        row, col = divmod(int(index), self.scan_shape[1])
        return self.data.read(scan_region=(row, row + 1, col, col + 1))[0, 0].cpu().numpy()

    def mean_dp(self):
        total_t = torch.zeros(self.det_shape, device=self.device)
        for _, _, _, block_t in self._blocks():
            total_t += block_t.sum((0, 1))
        return total_t / self.n_frames

    def masked_sum(self, mask):
        mask_t = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if self._native is not None:
            valid = getattr(self.data, 'valid', None)
            if valid is not None:
                mask_t = mask_t & valid
            # Native ANS kernels reduce on-device; only a small 2D product crosses
            # the host boundary. No decoded measurement cube is constructed.
            image = self._native.masked_sum(mask_t.cpu().numpy())
            r0, r1, c0, c1 = self.data._detector_region
            return torch.as_tensor(image[r0:r1, c0:c1], device=self.device)
        image_t = torch.empty(self.scan_shape, device=self.device)
        for row, col, stop, block_t in self._blocks():
            image_t[row, col:stop] = block_t[..., mask_t].sum(-1)[0]
        return image_t

    def masked_sum_native(self, mask, *, out=None):
        image_t = self.masked_sum(mask)
        if out is not None:
            out.copy_(image_t)
            return out
        return image_t

    def reduce_frames(self, indices, reduce='mean'):
        if reduce not in ('mean', 'sum', 'max'):
            raise ValueError(f"Unknown reduction {reduce!r}; use mean, sum or max.")
        total_t = None
        count = 0
        for index in indices:
            row, col = divmod(int(index), self.scan_shape[1])
            value_t = self.data.read(scan_region=(row, row + 1, col, col + 1))[0, 0].float()
            total_t = value_t.clone() if total_t is None else (
                torch.maximum(total_t, value_t) if reduce == 'max' else total_t + value_t
            )
            count += 1
        if not count:
            raise ValueError('Select at least one scan position.')
        return (total_t / count if reduce == 'mean' else total_t).cpu().numpy()

    def close(self):
        # Borrowed data outlives detector queries.
        pass
