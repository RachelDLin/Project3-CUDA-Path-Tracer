#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"

// ===============================
// REQUIRED stubs when NO_STB is set
// ===============================

namespace tinygltf {

    bool LoadImageData(Image*,
        int,
        std::string*,
        std::string*,
        int,
        int,
        const unsigned char*,
        int,
        void*) {
        // We intentionally do not load images from glTF
        return true;
    }

    bool WriteImageData(const std::string*,
        const std::string*,
        const Image*,
        bool,
        const FsCallbacks*,
        const URICallbacks*,
        std::string*,
        void*) {
        // We intentionally do not write images
        return true;
    }

} // namespace tinygltf
