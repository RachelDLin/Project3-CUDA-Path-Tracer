#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "tiny_gltf.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"


using namespace std;
using namespace utilityCore;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}


void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    std::filesystem::path scenePath(jsonName);
    std::filesystem::path sceneDir = scenePath.parent_path();
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.roughness = p["ROUGHNESS"];
            const auto& specCol = p["SPECULAR_COLOR"];
            newMaterial.specular = glm::vec3(specCol[0], specCol[1], specCol[2]);
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);

            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p["IOR"];

            newMaterial.hasReflective = 1.0f;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];

        if (type == "custom_obj") {
            std::filesystem::path objRelPath = p["PATH"].get<std::filesystem::path>();
            std::filesystem::path objFullPath = std::filesystem::absolute(sceneDir / objRelPath).lexically_normal();

            std::vector<Triangle> objTris;
            loadFromOBJ(objFullPath.string(), objTris);

            int matId = MatNameToID[p["MATERIAL"]];

            glm::vec3 trans(p["TRANS"][0], p["TRANS"][1], p["TRANS"][2]);
            glm::vec3 rot(p["ROTAT"][0], p["ROTAT"][1], p["ROTAT"][2]);
            glm::vec3 scale(p["SCALE"][0], p["SCALE"][1], p["SCALE"][2]);

            glm::mat4 T = utilityCore::buildTransformationMatrix(trans, rot, scale);
            glm::mat4 N = glm::inverseTranspose(T);

            for (auto& tri : objTris) {
                tri.materialId = matId;

                auto transformVertex = [&](Vertex& v) {
                    glm::vec3 p = float3ToGlm(v.position);
                    glm::vec3 n = float3ToGlm(v.normal);

                    p = glm::vec3(T * glm::vec4(p, 1.0f));
                    n = glm::normalize(glm::vec3(N * glm::vec4(n, 0.0f)));

                    v.position = glmToFloat3(p);
                    v.normal = glmToFloat3(n);
                };

                transformVertex(tri.v1);
                transformVertex(tri.v2);
                transformVertex(tri.v3);
                
                triangles.push_back(tri);
            }
            printf("Loaded %s\n", objFullPath.string().c_str());
        }
        else if (type == "custom_gltf") {
            std::filesystem::path gltfRelPath = p["PATH"].get<std::filesystem::path>();
            std::filesystem::path gltfFullPath = std::filesystem::absolute(sceneDir / gltfRelPath).lexically_normal();

            std::vector<Triangle> gltfTris;
            loadFromGLTF(gltfFullPath.string(), gltfTris);

            int matId = MatNameToID[p["MATERIAL"]];

            glm::vec3 trans(p["TRANS"][0], p["TRANS"][1], p["TRANS"][2]);
            glm::vec3 rot(p["ROTAT"][0], p["ROTAT"][1], p["ROTAT"][2]);
            glm::vec3 scale(p["SCALE"][0], p["SCALE"][1], p["SCALE"][2]);

            glm::mat4 T = utilityCore::buildTransformationMatrix(trans, rot, scale);
            glm::mat4 N = glm::inverseTranspose(T);

            for (auto& tri : gltfTris) {
                tri.materialId = matId;

                auto transformVertex = [&](Vertex& v) {
                    glm::vec3 p = float3ToGlm(v.position);
                    glm::vec3 n = float3ToGlm(v.normal);

                    p = glm::vec3(T * glm::vec4(p, 1.0f));
                    n = glm::normalize(glm::vec3(N * glm::vec4(n, 0.0f)));

                    v.position = glmToFloat3(p);
                    v.normal = glmToFloat3(n);
                    };

                transformVertex(tri.v1);
                transformVertex(tri.v2);
                transformVertex(tri.v3);

                triangles.push_back(tri);
            }
            printf("Loaded %s\n", gltfFullPath.string().c_str());
        }
        else {
            Geom newGeom;
            if (type == "cube")
            {
                newGeom.type = CUBE;
            }
            else if (type == "sphere")
            {
                newGeom.type = SPHERE;
            }
            newGeom.materialId = MatNameToID[p["MATERIAL"]];

            newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
            newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
            newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

            geoms.push_back(newGeom);
        }
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);
    camera.aperture = cameraData["APERTURE"];
    camera.focalDistance = glm::length(camera.lookAt - camera.position);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::loadFromOBJ(
    const std::string& objName,
    std::vector<Triangle>& outTriangles
) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ret = tinyobj::LoadObj(
        &attrib,
        &shapes,
        &materials,
        &warn,
        &err,
        objName.c_str(),
        /* mtl_basedir = */ nullptr,
        /* triangulate = */ true
    );

    printf("here\n");

    if (!warn.empty()) {
        std::cout << "TinyOBJ warning: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "TinyOBJ error: " << err << std::endl;
    }
    if (!ret) {
        throw std::runtime_error("Failed to load OBJ file: " + objName);
    }

    // Iterate over shapes
    for (const auto& shape : shapes) {
        size_t indexOffset = 0;

        // Each face is guaranteed to be a triangle
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) {
                // Should not happen with triangulate=true
                indexOffset += fv;
                continue;
            }

            Triangle tri;
            
            Vertex vertices[3];

            // Material ID (-1 if none)
            tri.materialId = shape.mesh.material_ids[f];

            // Build the 3 vertices
            for (int v = 0; v < 3; v++) {
                tinyobj::index_t idx = shape.mesh.indices[indexOffset + v];

                Vertex vert{};

                // Position
                vert.position = make_float3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]
                );

                // Normal
                if (idx.normal_index >= 0) {
                    vert.normal = make_float3(
                        attrib.normals[3 * idx.normal_index + 0],
                        attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2]
                    );
                }
                else {
                    vert.normal = make_float3(0.0f, 0.0f, 0.0f);
                }

                // Texcoord/UV
                if (idx.texcoord_index >= 0) {
                    vert.texcoord = make_float2(
                        attrib.texcoords[2 * idx.texcoord_index + 0],
                        attrib.texcoords[2 * idx.texcoord_index + 1]
                    );
                }
                else {
                    vert.texcoord = make_float2(0.0f, 0.0f);
                }

                vertices[v] = vert;
            }
            tri.v1 = vertices[0];
            tri.v2 = vertices[1];
            tri.v3 = vertices[2];

            outTriangles.push_back(tri);
            indexOffset += 3;
        }
    }
}



// Convert tinygltf buffer data to float
inline float getAccessorValueFloat(const tinygltf::Model& model, const tinygltf::Accessor& accessor, size_t index) {
    const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buffer = model.buffers[view.buffer];
    const unsigned char* dataPtr = buffer.data.data() + view.byteOffset + accessor.byteOffset + index * accessor.ByteStride(view);
    float value;
    memcpy(&value, dataPtr, sizeof(float));
    return value;
}

// Convert tinygltf accessor to glm::vec3
glm::vec3 getVec3(const tinygltf::Model& model, const tinygltf::Accessor& accessor, size_t index) {
    float x = getAccessorValueFloat(model, accessor, index * 3 + 0);
    float y = getAccessorValueFloat(model, accessor, index * 3 + 1);
    float z = getAccessorValueFloat(model, accessor, index * 3 + 2);
    return glm::vec3(x, y, z);
}

// Transform all vertices of a glTF mesh primitive into Triangle vector
void loadMeshFromGLTF(const tinygltf::Mesh& mesh, const tinygltf::Model& model, glm::mat4 T, std::vector<Triangle>& outTriangles) {
    glm::mat4 N = glm::inverseTranspose(T); // For normals

    for (const auto& prim : mesh.primitives) {
        // Access positions, normals, texcoords
        const tinygltf::Accessor& posAccessor = model.accessors[prim.attributes.find("POSITION")->second];
        const tinygltf::Accessor& normalAccessor = prim.attributes.count("NORMAL") ? model.accessors[prim.attributes.find("NORMAL")->second] : posAccessor; // fallback
        const tinygltf::Accessor& uvAccessor = prim.attributes.count("TEXCOORD_0") ? model.accessors[prim.attributes.find("TEXCOORD_0")->second] : posAccessor; // fallback

        const tinygltf::Accessor& indexAccessor = model.accessors[prim.indices];

        for (size_t f = 0; f < indexAccessor.count; f += 3) {
            Triangle tri;
            tri.materialId = prim.material;

            unsigned int idx[3];
            for (int j = 0; j < 3; j++) {
                // Extract indices
                switch (indexAccessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                    idx[j] = reinterpret_cast<const unsigned short*>(model.buffers[model.bufferViews[indexAccessor.bufferView].buffer].data.data() + indexAccessor.byteOffset)[f + j];
                    break;
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                    idx[j] = reinterpret_cast<const unsigned int*>(model.buffers[model.bufferViews[indexAccessor.bufferView].buffer].data.data() + indexAccessor.byteOffset)[f + j];
                    break;
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                    idx[j] = reinterpret_cast<const unsigned char*>(model.buffers[model.bufferViews[indexAccessor.bufferView].buffer].data.data() + indexAccessor.byteOffset)[f + j];
                    break;
                }
            }

            Vertex vertices[3];
            for (int j = 0; j < 3; j++) {
                // Get data
                glm::vec3 p = getVec3(model, posAccessor, idx[j]);
                glm::vec3 n = getVec3(model, normalAccessor, idx[j]);
                glm::vec2 uv = uvAccessor.count > 0 ? glm::vec2(getAccessorValueFloat(model, uvAccessor, idx[j] * 2), getAccessorValueFloat(model, uvAccessor, idx[j] * 2 + 1)) : glm::vec2(0.0f);

                // Transform into world space
                p = glm::vec3(T * glm::vec4(p, 1.0f));
                n = glm::normalize(glm::vec3(N * glm::vec4(n, 0.0f)));

                vertices[j].position = make_float3(p.x, p.y, p.z);
                vertices[j].normal = make_float3(n.x, n.y, n.z);
                vertices[j].texcoord = make_float2(uv.x, uv.y);
            }

            tri.v1 = vertices[0];
            tri.v2 = vertices[1];
            tri.v3 = vertices[2];

            outTriangles.push_back(tri);
        }
    }
}

// Recursive node processing
void processNode(const tinygltf::Model& model, int nodeIndex, glm::mat4 parentTransform, std::vector<Triangle>& outTriangles) {
    const auto& node = model.nodes[nodeIndex];
    glm::mat4 T = parentTransform * utilityCore::buildTransformationMatrix(
        glm::vec3(node.translation.empty() ? 0.0f : node.translation[0],
            node.translation.empty() ? 0.0f : node.translation[1],
            node.translation.empty() ? 0.0f : node.translation[2]),
        glm::vec3(node.rotation.empty() ? 0.0f : node.rotation[0],
            node.rotation.empty() ? 0.0f : node.rotation[1],
            node.rotation.empty() ? 0.0f : node.rotation[2]),
        glm::vec3(node.scale.empty() ? 1.0f : node.scale[0],
            node.scale.empty() ? 1.0f : node.scale[1],
            node.scale.empty() ? 1.0f : node.scale[2])
    );

    // Process meshes
    if (node.mesh >= 0) {
        loadMeshFromGLTF(model.meshes[node.mesh], model, T, outTriangles);
    }

    // Recurse children
    for (int child : node.children) {
        processNode(model, child, T, outTriangles);
    }
}

void Scene::loadFromGLTF(const std::string& gltfFile, std::vector<Triangle>& outTriangles) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    loader.SetImageLoader(
        [](tinygltf::Image*, int, std::string*, std::string*, int, int, const unsigned char*, int, void*) {
            return true; // pretend image loaded
        },
        nullptr
    );


    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltfFile);
    if (!warn.empty()) std::cout << "TinyGLTF warning: " << warn << std::endl;
    if (!err.empty()) std::cerr << "TinyGLTF error: " << err << std::endl;
    if (!ret) throw std::runtime_error("Failed to load glTF: " + gltfFile);

    for (int i = 0; i < model.scenes[model.defaultScene].nodes.size(); i++) {
        processNode(model, model.scenes[model.defaultScene].nodes[i], glm::mat4(1.0f), outTriangles);
    }

    std::cout << "Loaded glTF " << gltfFile << " with " << outTriangles.size() << " triangles\n";
}
