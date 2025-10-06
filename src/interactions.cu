#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}


__device__ void coordinateSystem(const glm::vec3& v1, glm::vec3& v2, glm::vec3& v3) {
    if (abs(v1.x) > abs(v1.y))
        v2 = glm::vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        v2 = glm::vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    v3 = cross(v1, v2);
}

__device__ glm::mat3 LocalToWorld(glm::vec3 nor) {
    glm::vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

__device__ glm::mat3 WorldToLocal(glm::vec3 nor) {
    return transpose(LocalToWorld(nor));
}

__device__ glm::vec3 squareToDiskConcentric(glm::vec2 xi) {
    // Peter Shirley's warping method (concentric circles)

    // remap coords from 0..1 to -1..1
    glm::vec2 remappedSample = 2.0f * xi - 1.0f;

    float xi1 = remappedSample.x;
    float xi2 = remappedSample.y;

    // compute polar coords
    float radius, angle;
    if (abs(xi1) > abs(xi2)) {
        radius = xi1;

        if (xi1 == 0) {
            angle = 0;
        }
        else {
            angle = (PI / 4.0f) * (xi2 / xi1);
        }
    }
    else {
        radius = xi2;

        if (xi2 == 0) {
            angle = 0;
        }
        else {
            angle = (PI / 2.0f) - (PI / 4.0f) * (xi1 / xi2);
        }
    }

    // convert to cartesian coords
    float x = radius * cos(angle);
    float y = radius * sin(angle);

    return glm::vec3(x, y, 0);
}

__device__ glm::vec3 squareToHemisphereCosine(glm::vec2 xi) {
    float xi1 = xi.x;
    float xi2 = xi.y;

    // mapping to disk (polar coords)
    glm::vec3 diskSample = squareToDiskConcentric(xi);

    // convert to cartesian coords
    float z = sqrt(1.0f - diskSample.x * diskSample.x - diskSample.y * diskSample.y);

    return glm::vec3(diskSample.x, diskSample.y, z);
}

__device__ float squareToHemisphereCosinePDF(glm::vec3 sample) {
    return sample.z / PI;
}

__device__ glm::vec3 f_diffuse(glm::vec3 albedo) {
    // for lambertian surfaces, f(p, wo, wi) = R/pi
    // for cosine-weighted hemisphere sampling, PDF is cos(theta)/pi
    return albedo / PI;
}

__device__ glm::vec3 Sample_f_diffuse(const glm::vec3 albedo, 
    const glm::vec3 nor,
    glm::vec3& wiW, 
    float& pdf,  
    thrust::default_random_engine& rng) {
    // Make sure you set wiW to a world-space ray direction,
    // since wo is in tangent space. You can use
    // the function LocalToWorld() in the "defines" file
    // to easily make a mat3 to do this conversion.

    // sample an incoming ray direction
    thrust::uniform_real_distribution<float> u01(0, 1);
    const glm::vec2 xi = glm::vec2(u01(rng), u01(rng));
    wiW = squareToHemisphereCosine(xi);

    // get pdf (probability of dir wiW being chosen as a sample)
    pdf = squareToHemisphereCosinePDF(wiW);

    // convert wiW to world space
    wiW = glm::normalize(LocalToWorld(nor) * wiW);

    // compute diffuse color
    return f_diffuse(albedo);
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    // generate a new ray direction using the cosine-wtd scatter fn calculateRandomDirectionInHemisphere
    glm::vec3 newDir = calculateRandomDirectionInHemisphere(normal, rng);

    // update ray origin and direction
    // small offset to avoid self-intersection
    float offset = 0.0001f;
    pathSegment.ray.origin = intersect + offset * normal;
    pathSegment.ray.direction = glm::normalize(newDir);

    // cosine weighting
    float cosine = glm::dot(normal, newDir);
    float pdf = glm::cos(glm::acos(cosine)) / PI;

    // check that probabilities are btwn 0 and 1
    if (pdf <= 0.f || pdf > 1.f) {
        pathSegment.color = glm::vec3(0.f);
    }
    else {
        // multiply path color (light color) by the diffuse albedo (material color)
        glm::vec3 bsdf = m.color / PI;
        pathSegment.color = bsdf * glm::abs(cosine) / pdf;
    }    
}

__device__ void scatterRay_diffuse(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{

    glm::vec3 wiW;
    float pdf;
    glm::vec3 bsdf;

    // sample diffuse bsdf and initialize direction and pdf
    bsdf = Sample_f_diffuse(m.color, normal, wiW, pdf, rng);

    // get ray
    Ray& ray = pathSegment.ray;

    // update ray origin and direction
    // small offset to avoid self-intersection
    float offset = 0.0001f;
    ray.origin = intersect + offset * normal;
    ray.direction = glm::normalize(wiW);

    // check that probabilities are btwn 0 and 1
    if (pdf <= 0.f || pdf > 1.f) {
        pathSegment.color = glm::vec3(0.f);
    }
    else {
        // obstructed?
        float V = 1.f;

        // lambertian term (cosine factor)
        float lambert = glm::abs(glm::dot(normal, wiW));

        // update throughput: throughput_k = throughput_{k-1} * multiplier
        // multiplier = BSDF * cosine factor * transmittance (if the medium absorbs light) / pdf(wi)
        glm::vec3 multiplier = bsdf * V * lambert / pdf;

        // throughput
        pathSegment.color *= multiplier;
    }    
}
