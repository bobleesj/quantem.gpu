# Third-Party Notices

`quantem.gpu` is distributed under the MIT License. The package also includes
small portions of third-party source code listed below.

## NVIDIA nvCOMP CUDA LZ4 Decompression Kernel

File:

- `src/quantem/gpu/io/bitshuffle.py`

The CUDA LZ4 decompression kernel embedded in `_CUDA_LZ4_SOURCE` carries this
upstream notice:

```text
LZ4 decompression kernel extracted from NVIDIA nvcomp
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
BSD-3-Clause License
```

This distribution preserves the NVIDIA copyright notice and license
classification from the source header. The QuantEM project is not sponsored or
endorsed by NVIDIA Corporation.

BSD-3-Clause License text:

```text
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of NVIDIA CORPORATION nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
