#ifndef PANORAMARENDERER_H
#define PANORAMARENDERER_H

#include <GLES3/gl3.h>
#include <android/asset_manager.h>
#include "opencv2/opencv.hpp"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "Sphere.h"

class PanoramaRenderer {
public:
    PanoramaRenderer(AAssetManager* assetManager,std::string filepath);
    ~PanoramaRenderer();

    void onSurfaceCreated();
    void onDrawFrame();
    void onSurfaceChanged(int width, int height);

    void handleTouchDrag(float deltaX, float deltaY);
    void handlePinchZoom(float scaleFactor);

private:
    GLuint loadShader(GLenum type, const char *shaderSrc);
    GLuint createProgram(const char *vertexSrc, const char *fragmentSrc);
    GLuint loadTexture(const char *assetPath);

    GLuint shaderProgram;
    GLuint texture;
    GLuint vao, vboVertices, vboTexCoords,vboIndices;
    SphereData* sphereData;
    AAssetManager* assetManager;
    std::string sharePath;

    glm::mat4 projection;
    glm::mat4 view;
    float rotationX;
    float rotationY;
    float zoom;
};

#endif // PANORAMARENDERER_H
