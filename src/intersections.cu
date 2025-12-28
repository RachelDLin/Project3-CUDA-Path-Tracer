#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }
    
    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(
    const Triangle& tri,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside) {

    // get vtx info
    const Vertex& v1 = tri.v1;
    const Vertex& v2 = tri.v2;
    const Vertex& v3 = tri.v3;

    // --- Moller-Trumbore ray-triangle intersection ---

    // compute face normal
    glm::vec3 e1 = v2.position - v1.position;
    glm::vec3 e2 = v3.position - v1.position;
    glm::vec3 triNor = glm::normalize(glm::cross(e1, e2));

    // check if ray and triangle are parallel (no intersection)
    if (glm::dot(r.direction, triNor) < 0.00001) {
        return -1;
    }
    else {

        // find location where ray intersects tri plane
        // Moller-Trumbore: r.origin + (t * r.direction) = v1.pos + u * e1 + v * e2;
        // O + t*D = p1 + u*e1 + v*e2
        // O - p1 = u*e1 + v*e2 - t*D
        // [[e1 e2 -D] [u v t]^T = O - p1
        // Cramer's rule: for system of equations Ax = b, xi = det(Ai) / det(A) where Ai is A w/ the ith column replaced by b
        // A = [e1 e2 -D]
        // x = [u v t]^T
        // b = O - p1
        glm::vec3 D = r.direction;
        glm::vec3 O = r.origin;
        glm::vec3 b = O - v1.position;

        // solve for t
        float detA = glm::dot(e1, glm::cross(D, e2));
        if (fabs(detA) < 1e-8) return -1;

        float detAi = glm::dot(e2, glm::cross(b, e1));
        if (fabs(detAi) < 1e-8) return -1;

        float t = detA / detAi;

        // compute intersection pt
        intersectionPoint = O + D * t;

        // solve for barycentric coords
        glm::vec3 d0 = intersectionPoint - v1.position;
        float d00 = dot(e1, e2);
        float d01 = dot(e1, e2);
        float d11 = dot(e2, e2);
        float d20 = dot(d0, e1);
        float d21 = dot(d0, e2);

        float denom = d00 * d11 - d01 * d01;

        float b1 = (d11 * d20 - d01 * d21) / denom;
        float b2 = (d00 * d21 - d01 * d20) / denom;
        float b3 = 1.0f - b1 - b2;

        // get intersection surface normal
        normal = v1.normal * b1 + v2.normal * b2 + v3.normal * b3;

        // get intersection uv
        uv = v1.texcoord * b1 + v2.texcoord * b2 + v3.texcoord * b3;

        // check side that ray hits
        if (glm::dot(triNor, r.direction) < 0) {
            // front face
            outside = true;
        }
        else {
            // back face
            outside = false;
        }

        // return distance from origin to intersection pt
        return glm::length(intersectionPoint - O);
    }
    return 0;
}