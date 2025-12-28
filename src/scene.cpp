#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace std;
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

                auto& v1 = tri.v1;
                v1.position = glm::vec3(T * glm::vec4(v1.position, 1.0f));
                v1.normal = glm::normalize(glm::vec3(N * glm::vec4(v1.normal, 0.0f)));

                auto& v2 = tri.v2;
                v2.position = glm::vec3(T * glm::vec4(v2.position, 1.0f));
                v2.normal = glm::normalize(glm::vec3(N * glm::vec4(v2.normal, 0.0f)));

                auto& v3 = tri.v3;
                v3.position = glm::vec3(T * glm::vec4(v3.position, 1.0f));
                v3.normal = glm::normalize(glm::vec3(N * glm::vec4(v3.normal, 0.0f)));
                
                triangles.push_back(tri);
            }
            printf("Loaded %s\n", objFullPath.string().c_str());
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
                vert.position = glm::vec3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]
                );

                // Normal
                if (idx.normal_index >= 0) {
                    vert.normal = glm::vec3(
                        attrib.normals[3 * idx.normal_index + 0],
                        attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2]
                    );
                }
                else {
                    vert.normal = glm::vec3(0.0f);
                }

                // Texcoord/UV
                if (idx.texcoord_index >= 0) {
                    vert.texcoord = glm::vec2(
                        attrib.texcoords[2 * idx.texcoord_index + 0],
                        attrib.texcoords[2 * idx.texcoord_index + 1]
                    );
                }
                else {
                    vert.texcoord = glm::vec2(0.0f);
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

