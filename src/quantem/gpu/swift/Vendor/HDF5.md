# Vendored HDF5

`CHDF5.xcframework` contains the official HDF5 1.14.6 C library from
`HDFGroup/hdf5` tag `hdf5-1.14.6` (commit
`7bf340440909d468dbb3cf41f0ea0d87f5050cea`). It is a static Apple-silicon
macOS build with a macOS 14 deployment target. High-level APIs, command-line
tools, tests, C++, and Fortran are disabled; zlib support remains enabled for
small compressed metadata datasets.

The archive was built from a generic `/tmp/quantem-hdf5-1.14.6.*` tree. HDF5
embeds its configure summary in the static library, so the temporary generated
`configure` script was changed to report `quantem-build` and
`Apple arm64 macOS 14+` instead of the operator account and build-host name.
These two provenance strings do not change HDF5 code. The archive was then
configured with:

```sh
perl -pi -e 's/CONFIG_USER="`whoami`@`hostname`"/CONFIG_USER="quantem-build"/' configure
perl -pi -e 's/UNAME_INFO=`uname -a`/UNAME_INFO="Apple arm64 macOS 14+"/' configure
MACOSX_DEPLOYMENT_TARGET=14.0 \
CFLAGS="-O3 -DNDEBUG -fvisibility=hidden -mmacosx-version-min=14.0" \
../configure \
  --prefix=/opt/quantem/hdf5 \
  --disable-shared \
  --enable-static \
  --disable-hl \
  --disable-tests \
  --disable-tools \
  --with-zlib=/usr
```

After `make -j` and `make DESTDIR=stage install`, the artifact was created with:

```sh
xcodebuild -create-xcframework \
  -library stage/opt/quantem/hdf5/lib/libhdf5.a \
  -headers stage/opt/quantem/hdf5/include \
  -output CHDF5.xcframework
```

The SHA-256 checksum of `macos-arm64/libhdf5.a` is
`f13378e06be438a77bf233c74af942e973f8c922f7c52bfcb5b2e07a74ad258f`.
Before publishing, run `strings` over the archive and reject build-user,
hostname, home-directory, password, or repository-path matches.
The upstream license is bundled with the `Native4DSTEMIO` resource bundle.
