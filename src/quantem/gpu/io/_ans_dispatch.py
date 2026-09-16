"""Resident representation conversion for encoded count sources."""

from copy import deepcopy

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
        loaded.representation is DataRepresentation.ENCODED
        and target is DataRepresentation.PACKED
    ):
        convert = getattr(source, "to_packed", None)
        if convert is None:
            raise NotImplementedError(
                "Direct encoded-to-packed conversion is not qualified for this backend yet."
            )
        output = convert()
    elif (loaded.representation is DataRepresentation.DENSE
          and target is DataRepresentation.PACKED):
        from .backends.cuda._ans import CudaPackedResidentCounts

        output = CudaPackedResidentCounts.from_array(source, loaded.shape)
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
