#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromOBJ(const std::string& objName, std::vector<Triangle>& outTriangles);
    void loadFromGLTF(const std::string& gltfName, std::vector<Triangle>& outTriangles);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    std::vector<Triangle> triangles;
};
