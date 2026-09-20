"""Metal-backed PyTorch queries with parallel literal bit-lane access."""

from functools import cache
from pathlib import Path

import numpy as np

from ._streamed import MPSStreamedCounts


@cache
def _kernels():
    from .packed import _metal_module

    metal = _metal_module()
    device = metal.MTLCreateSystemDefaultDevice()
    root = Path(__file__).parent / "kernels"
    source = (
        (root / "streamed_counts.msl")
        .read_text()
        .split("kernel void streamed_counts_encode")[0]
    )
    source += (root / "float_ans.msl").read_text()
    options = metal.MTLCompileOptions.alloc().init()
    options.setFastMathEnabled_(False)
    library, error = device.newLibraryWithSource_options_error_(source, options, None)
    if library is None:
        raise RuntimeError(f"Float ANS Metal compilation failed: {error}")
    pipelines = []
    for name in ("direct", "entropy", "selected_entropy", "detector"):
        function = library.newFunctionWithName_("float_ans_" + name)
        pipeline, error = device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if pipeline is None:
            raise RuntimeError(f"Float ANS Metal pipeline failed: {error}")
        pipelines.append(pipeline)
    return pipelines


class MPSFloatLanes(MPSStreamedCounts):
    """Write the original bit lanes directly into caller-owned Metal storage."""

    def _encode_decode(
        self, command, chunk, local_first, count, output, output_offset_bytes=0
    ):
        direct, entropy = _kernels()[:2]
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(direct)
        for index, buffer in enumerate(chunk.buffers):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.setBuffer_offset_atIndex_(output, output_offset_bytes, 3)
        parameters = np.asarray([local_first, count], np.uint32).tobytes()
        encoder.setBytes_length_atIndex_(parameters, len(parameters), 4)
        self._dispatch_threads(encoder, count * 32768)
        encoder.endEncoding()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(entropy)
        for index, buffer in enumerate((*chunk.buffers, self._decoding)):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.setBuffer_offset_atIndex_(output, output_offset_bytes, 4)
        encoder.setBuffer_offset_atIndex_(self._errors, 0, 5)
        parameters = np.asarray([chunk.scans, local_first, count], np.uint32).tobytes()
        encoder.setBytes_length_atIndex_(parameters, len(parameters), 6)
        self._dispatch_threads(encoder, 32768)
        encoder.endEncoding()

    def float_detector(self, mask, background=None):
        """Submit ordered selected decode/reduction pairs with one completion wait."""
        import ctypes
        import objc
        import torch
        from ._streamed import _upload
        from .packed import _allocate_shared, _release, _complete

        entropy, reduce = _kernels()[2:]
        output = torch.empty(self.shape[:2], dtype=torch.float32, device="mps")
        torch.mps.synchronize()
        target = objc.objc_object(
            c_void_p=ctypes.c_void_p(output.untyped_storage().data_ptr())
        )
        dark = (
            target
            if background is None
            else objc.objc_object(
                c_void_p=ctypes.c_void_p(background.untyped_storage().data_ptr())
            )
        )
        selected = scratch = None
        try:
            selected = _upload(
                self._device, self._metal, mask.astype(np.uint8), "Float detector mask"
            )
            scratch = _allocate_shared(
                self._device,
                self._metal,
                max(c.scans for c in self.chunks) * 65536,
                "Bounded float entropy scratch",
            )
            self._clear_errors()
            command = self._queue.commandBuffer()
            for chunk in self.chunks:
                encoder = command.computeCommandEncoder()
                encoder.setComputePipelineState_(entropy)
                for i, buffer in enumerate(
                    (*chunk.buffers, self._decoding, scratch, self._errors, selected)
                ):
                    encoder.setBuffer_offset_atIndex_(buffer, 0, i)
                parameters = np.uint32(chunk.scans).tobytes()
                encoder.setBytes_length_atIndex_(parameters, len(parameters), 7)
                self._dispatch_threads(encoder, 32768)
                encoder.endEncoding()
                encoder = command.computeCommandEncoder()
                encoder.setComputePipelineState_(reduce)
                for i, buffer in enumerate(
                    (*chunk.buffers, scratch, selected, dark, target)
                ):
                    encoder.setBuffer_offset_atIndex_(buffer, 0, i)
                parameters = np.asarray(
                    [chunk.first, background is not None], np.uint32
                ).tobytes()
                encoder.setBytes_length_atIndex_(parameters, len(parameters), 7)
                encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                    self._metal.MTLSizeMake(chunk.scans, 1, 1),
                    self._metal.MTLSizeMake(128, 1, 1),
                )
                encoder.endEncoding()
            _complete(command, "Float ANS detector")
            self._check_errors()
            return output
        finally:
            _release(scratch)
            _release(selected)
