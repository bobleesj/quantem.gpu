"""One public loader boundary for count-ANS files and resident conversion."""

import hashlib
import math
from copy import deepcopy
from contextlib import ExitStack

import numpy as np

from ._ans import ANSFile
from .models import FourDSTEMData, _release_owned_storage
from .representation import DataRepresentation


def _convert_resident(loaded, representation):
    """Convert explicitly without changing counts or releasing the caller's input."""
    target = DataRepresentation.parse(representation)
    if target is loaded.representation:
        return loaded
    source = loaded.data
    metadata = deepcopy(loaded.metadata)
    if (
        loaded.representation is DataRepresentation.ANS
        and target is DataRepresentation.PACKED
    ):
        convert = getattr(source, "to_packed", None)
        if convert is None:
            raise NotImplementedError(
                "Direct ANS-to-packed conversion is not qualified for this backend yet."
            )
        output = convert()
    else:
        raise NotImplementedError(
            f"{loaded.representation.value}-to-{target.value} resident conversion "
            "is not implemented yet. No dense expansion or CPU fallback was performed."
        )
    try:
        metadata.update(
            representation=target.value,
            physical_resident_bytes=output.nbytes,
            conversion_from=loaded.representation.value,
            resident_profile="block-column-bitpacked-u32-v1",
        )
        return FourDSTEMData(output, metadata)
    except BaseException as error:
        _release_owned_storage(output, failure=error)
        raise


def _load_ans(source, *, backend, representation, expected_sha256, device):
    """Authenticate/map source, validate exact streams, and retain selected encoding."""
    from .backends import resolve_backend

    backend = resolve_backend(backend)
    target = DataRepresentation.parse(representation)
    if backend == "cpu" and target is not DataRepresentation.DENSE:
        raise NotImplementedError(
            "CPU reference loading requires representation='dense'; GPU residency needs CUDA or MPS."
        )
    if device is not None and backend != "cuda":
        raise ValueError("Explicit ANS device selection requires backend='cuda'; omit device for CPU or MPS.")
    if target is DataRepresentation.DENSE and backend != "cpu":
        raise NotImplementedError(
            "ANS-to-dense GPU materialization is not qualified yet; load as 'ans' or 'packed'."
        )
    owner = None
    with ANSFile(source, expected_sha256=expected_sha256) as encoded, ExitStack() as contexts:
        metadata = deepcopy(encoded.metadata)
        source_shape = metadata.get("source_shape", encoded.shape)
        source_dtype = metadata.get("source_dtype", encoded.dtype.name)
        metadata.update(
            backend=backend,
            representation=DataRepresentation.ANS.value,
            residency="host" if backend == "cpu" else "device",
            source_shape=source_shape,
            working_shape=encoded.shape,
            source_dtype=source_dtype,
            working_dtype=encoded.dtype.name,
            scan_shape=encoded.shape[:2],
            detector_shape=encoded.shape[2:],
            dtype=encoded.dtype.name,
            n_frames=math.prod(encoded.shape[:2]),
            source_logical_tensor_bytes=metadata.get(
                "source_logical_tensor_bytes",
                math.prod(source_shape) * np.dtype(source_dtype).itemsize,
            ),
            working_logical_tensor_bytes=encoded.logical_nbytes,
            container_bytes=encoded.file_bytes,
            encoded_bytes=encoded.encoded_nbytes,
            storage_schema=encoded.manifest["schema"],
            container_encoding="ans",
            resident_profile=encoded.manifest["codec"],
            container_path=str(encoded.path.resolve()),
            container_sha256=expected_sha256,
            container_authentication="external-sha256"
            if expected_sha256
            else "internal-checksums-only",
            declared_logical_sha256=encoded.manifest["logical_sha256"],
            logical_count_hash_verified=False,
            lossless_exact=metadata.get("lossless_exact", True),
            file_counts_exact=True,
            scan_bin=metadata.get("scan_bin", 1),
            detector_bin=metadata.get("detector_bin", 1),
            crop=metadata.get("crop"),
            detector_mask_policy="preserve-stored-counts",
        )
        try:
            if backend == "cpu":
                # Explicit reference use only. Fill final storage block by block,
                # never concatenate a redundant list of complete dense blocks.
                owner = np.empty(encoded.shape, dtype=encoded.dtype)
                frames = owner.reshape(-1, *encoded.shape[2:])
                logical_digest = hashlib.sha256()
                for block, first in enumerate(
                    range(0, len(frames), encoded.block_frames)
                ):
                    values = encoded.decode_block_reference(block)
                    frames[first : first + len(values)] = values
                    logical_digest.update(memoryview(values).cast("B"))
                if logical_digest.hexdigest() != encoded.manifest["logical_sha256"]:
                    raise ValueError(
                        "Decoded ANS counts do not match the declared logical digest."
                    )
                metadata["logical_count_hash_verified"] = True
                metadata["representation"] = "dense"
                metadata["resident_profile"] = "native-dense"
            elif backend == "cuda":
                from .backends.cuda._ans import CudaANSResidentCounts

                import cupy as cp

                selected_device = cp.cuda.runtime.getDevice() if device is None else int(device)
                contexts.enter_context(cp.cuda.Device(selected_device))
                owner = CudaANSResidentCounts(**encoded.runtime_arguments())
            else:
                from .backends.mps._ans import MPSANSResidentCounts

                owner = MPSANSResidentCounts(**encoded.runtime_arguments())
            metadata["physical_resident_bytes"] = owner.nbytes
            encoded.assert_unchanged()
            loaded = FourDSTEMData(owner, metadata)
            if target is DataRepresentation.PACKED:
                converted = _convert_resident(loaded, target)
                try:
                    loaded.close()
                except BaseException as error:
                    _release_owned_storage(converted.data, failure=error)
                    raise
                return converted
            return loaded
        except BaseException as error:
            if owner is not None:
                _release_owned_storage(owner, failure=error)
            raise
