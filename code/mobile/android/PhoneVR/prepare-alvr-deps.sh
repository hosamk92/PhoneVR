if [ "$1" == "nogvr" ]; then
nogvr=true
shift
fi

rustup target add \
    aarch64-linux-android \
    armv7-linux-androideabi \
    x86_64-linux-android \
    i686-linux-android

# # Download ALVR source
# rm -r "ALVR"
# curl -sLS "https://github.com/alvr-org/ALVR/archive/refs/heads/master.zip" > download.zip
# # curl -sLS "https://github.com/ShootingKing-AM/ALVR/archive/refs/heads/phonevr.zip" > download.zip
# unzip download.zip
# rm download.zip
# mv ALVR-master ALVR
# # mv ALVR-phonevr ALVR

pushd ALVR
# ALVR >= 20.9 needs its own openvr submodule (headers used by alvr_session's build script)
git submodule update --init --recursive || exit 1
# Backport of ALVR a915efcf (#3104): strict GLES drivers (most phones) reject the staging shader
# without a default float precision and the client panics when the stream starts
STAGING_FRAG="alvr/graphics/resources/staging_fragment.glsl"
grep -q "precision mediump float" "$STAGING_FRAG" ||
    sed -i '2s/$/\n\nprecision mediump float;/' "$STAGING_FRAG" || exit 1
# Keep ALVR's own Cargo.lock (cargo update pulls untested crate versions)
cargo xtask prepare-deps --platform android $@
# ALVR 20.14 passes "-p <api level>" to cargo-ndk, which cargo-ndk 4.x reads as "--package"
cargo install cargo-ndk --version 3.5.4 --locked --force || exit 1
popd

rm -r cardboard

# Download sdk source$
CARB_REPO_NAME="cardboard-master"
rm -r "${CARB_REPO_NAME}"
# curl -sLS "https://github.com/googlevr/cardboard/archive/refs/heads/master.zip" > download.zip
curl -sLS "https://github.com/nift4/cardboard/archive/refs/heads/master.zip" > download.zip
unzip download.zip
rm download.zip

# Build sdk
pushd "${CARB_REPO_NAME}"
./gradlew sdk:assembleRelease -Parm64-v8a
popd

# Prepare files
mkdir cardboard
mv "${CARB_REPO_NAME}/sdk/build/outputs/aar/sdk-release.aar" cardboard/cardboard-sdk.aar
cp "${CARB_REPO_NAME}/sdk/include/cardboard.h" cardboard/cardboard.h
rm -r "${CARB_REPO_NAME}"

rm -r "gvr-android-sdk-1.200"
if [ ! $nogvr ]; then
curl -sLS "https://github.com/googlevr/gvr-android-sdk/releases/download/v1.200/gvr-android-sdk-1.200.zip" > download.zip
unzip download.zip
rm download.zip
fi

