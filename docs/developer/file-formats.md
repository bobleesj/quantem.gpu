# File formats and codecs

This section is for reader and writer implementers. To save a Python acquisition,
use [Save and share your data](../api/qem-python.md); no codec selection is needed.

## QEM: portable measurements and metadata

Read the specification in this order:

| Question | Authoritative page |
|---|---|
| What does the file contain, and what must round-trip? | [Container and scientific metadata](../api/qem-format.md) |
| How are integer counts and float bits encoded? | [Codec layouts](../api/qem-codecs.md) |
| Which vendor fields are interpreted or retained? | [Metadata mapping](../api/qem-metadata-mapping.md) |
| How do independent readers prove correctness? | [Conformance and interoperability](../api/qem-interoperability.md) |
| How does a qualified DM4 acquisition map to QEM? | [DM4 conversion details](../integrations/k3-dm4-qem.md) |

Container, metadata-schema, and codec versions identify different parts of the
file contract. Follow the identifiers declared in the file; a newer prose
specification does not imply that every backend supports every layout.

## Runtime storage and prepared formats

A device's resident representation and a portable acquisition file serve
different purposes. Python acquisition loading uses ANS; internal layouts and
prepared formats have their own callers and acceptance boundaries.

| Implementation task | Contract |
|---|---|
| Understand encoded, paired, packed, and dense layouts | [Count representations](../api/representations.md) |
| Implement the prepared packed-file layout | [Lossless Pack Format v1](../api/compact_4dstem_h5.md) |
| Produce that format from a native application | [Native producer](../api/native_lossless_pack_v1_producer.md) |
| Maintain the internal count-ANS container | [Count-ANS](count-ans.md) |
| Maintain paired resident counts | [Paired-count layout](paired-resident.md) |

The prepared packed format is not an alternate `io.load` mode for current
Python acquisitions. Use the original measurements to export a supported QEM
file. Dated codec experiments remain in [Historical records](../maintainer/index.md).
