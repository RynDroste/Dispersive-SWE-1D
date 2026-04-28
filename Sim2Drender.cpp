// ============================================================================
// Sim2Drender.cpp -- Modern OpenGL 3.3 Core + Dear ImGui frontend for the 2D
// dispersive shallow-water simulator (Sim2D.cu / Sim2D.h).
//
// Rendering strategy:
//   * Two GPU textures of size GRIDRESOLUTION_2D x GRIDRESOLUTION_2D:
//       - terrainTex (R32F, uploaded on reset)
//       - hTex       (R32F, uploaded every frame from sim.h[])
//       - htildeTex  (R32F, uploaded every frame, used as overlay color)
//   * One static (i, j)-grid mesh in vec2 attributes; both terrain and water
//     re-use it. The vertex shader looks up the height in the textures.
//   * Terrain is drawn opaque; water is drawn afterwards with alpha blending,
//     fading to fully transparent where h <= epsilon (dry cells).
//   * Per-frame normals are computed in the vertex shader from finite
//     differences of the height texture.
//
// Camera:
//   * Orbit camera (yaw/pitch/distance) around the centre of the grid.
//   * LMB-drag on the canvas    -> rotate
//   * RMB-drag on the canvas    -> pan
//   * Mouse wheel               -> zoom
//   * Middle-click on the canvas -> brush (add water; +Shift removes)
//
// UI (Dear ImGui):
//   * Reset terrain (flat / hill+island)
//   * Reset water   (const / dam / sloped / cosine droplet) + level slider
//   * Pause / single-step / SWE-only / substeps
//   * Brush size + strength
//   * Visual toggles (terrain mesh, water surface, htilde overlay,
//     wireframe water, vertical exaggeration)
//
// Build (CMake target: sim2d_viewer; see CMakeLists.txt):
//   cmake -S . -B build && cmake --build build --target sim2d_viewer
// ============================================================================

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "Sim2D.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

// ---------------------------------------------------------------------------
// Tiny inline math (column-major mat4, no glm dependency).
// ---------------------------------------------------------------------------
struct Vec3 { float x, y, z; };
static inline Vec3 V3(float x, float y, float z) { return {x, y, z}; }
static inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static inline float Dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline Vec3  Cross(Vec3 a, Vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
static inline float Length(Vec3 a) { return std::sqrt(Dot(a, a)); }
static inline Vec3  Normalize(Vec3 a) {
    float n = Length(a);
    return n > 1e-20f ? a * (1.f / n) : V3(0.f, 0.f, 0.f);
}

struct Mat4 { float m[16]; };

static Mat4 MatIdentity() {
    Mat4 r{}; r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.f; return r;
}
static Mat4 MatMul(const Mat4& A, const Mat4& B) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
        for (int rIdx = 0; rIdx < 4; ++rIdx) {
            float s = 0.f;
            for (int k = 0; k < 4; ++k) s += A.m[k * 4 + rIdx] * B.m[c * 4 + k];
            r.m[c * 4 + rIdx] = s;
        }
    return r;
}
static Mat4 MatPerspective(float fovyRad, float aspect, float znear, float zfar) {
    Mat4 r{};
    float f = 1.f / std::tan(fovyRad * 0.5f);
    r.m[0]  = f / aspect;
    r.m[5]  = f;
    r.m[10] = (zfar + znear) / (znear - zfar);
    r.m[11] = -1.f;
    r.m[14] = (2.f * zfar * znear) / (znear - zfar);
    return r;
}
static Mat4 MatLookAt(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = Normalize(center - eye);
    Vec3 s = Normalize(Cross(f, up));
    Vec3 u = Cross(s, f);
    Mat4 r = MatIdentity();
    r.m[0] = s.x; r.m[4] = s.y; r.m[ 8] = s.z;
    r.m[1] = u.x; r.m[5] = u.y; r.m[ 9] = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -Dot(s, eye);
    r.m[13] = -Dot(u, eye);
    r.m[14] =  Dot(f, eye);
    return r;
}

// ---------------------------------------------------------------------------
// GLSL shaders.
//   Both terrain and water use the same vertex shader; the difference is the
//   uniforms (uHeightOffset / which texture to add) and the fragment shader.
// ---------------------------------------------------------------------------
static const char* kVertSrc = R"GLSL(
#version 330 core
layout(location = 0) in vec2 aGrid;       // integer (i, j) in [0, N-1]

uniform mat4      uVP;
uniform sampler2D uTerrain;               // R32F
uniform sampler2D uWaterH;                // R32F (h)
uniform sampler2D uHTilde;                // R32F (only used by water frag)
uniform float     uTexel;                 // 1.0 / N
uniform float     uYScale;                // vertical exaggeration
uniform int       uIsWater;               // 0 = terrain, 1 = water surface

out vec3  vWorldPos;
out vec3  vNormal;
out vec2  vUV;
out float vH;
out float vHTilde;

float sampleTotal(vec2 uv) {
    float t = texture(uTerrain, uv).r;
    float h = uIsWater == 1 ? texture(uWaterH, uv).r : 0.0;
    return (t + h) * uYScale;
}

void main() {
    vec2 uv = (aGrid + 0.5) * uTexel;
    vUV = uv;

    float y  = sampleTotal(uv);
    vec3 P   = vec3(aGrid.x, y, aGrid.y);

    // Central differences for the surface normal (finite differences in xz).
    float yL = sampleTotal(uv - vec2(uTexel, 0.0));
    float yR = sampleTotal(uv + vec2(uTexel, 0.0));
    float yD = sampleTotal(uv - vec2(0.0, uTexel));
    float yU = sampleTotal(uv + vec2(0.0, uTexel));
    vec3 dx = vec3(2.0, yR - yL, 0.0);
    vec3 dz = vec3(0.0, yU - yD, 2.0);
    vNormal = normalize(cross(dz, dx));

    vH      = uIsWater == 1 ? texture(uWaterH,  uv).r : 0.0;
    vHTilde = uIsWater == 1 ? texture(uHTilde,  uv).r : 0.0;

    vWorldPos = P;
    gl_Position = uVP * vec4(P, 1.0);
}
)GLSL";

static const char* kFragSrc = R"GLSL(
#version 330 core
in vec3  vWorldPos;
in vec3  vNormal;
in vec2  vUV;
in float vH;
in float vHTilde;

uniform vec3  uLightDir;     // direction the light is travelling
uniform vec3  uViewPos;
uniform vec4  uBaseColor;
uniform int   uIsWater;
uniform float uHTildeGain;
uniform int   uShowHTilde;

out vec4 FragColor;

void main() {
    if (uIsWater == 1 && vH < 0.005) discard;        // dry cells

    vec3 N = normalize(vNormal);
    if (!gl_FrontFacing) N = -N;
    vec3 L = normalize(-uLightDir);
    vec3 V = normalize(uViewPos - vWorldPos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float amb  = 0.30;
    vec3  base = uBaseColor.rgb;
    vec3  col  = base * (amb + (1.0 - amb) * diff);

    if (uIsWater == 1) {
        float spec = pow(max(dot(N, H), 0.0), 96.0);
        col += vec3(0.85, 0.92, 1.0) * spec * 0.9;
        float fres = pow(1.0 - max(dot(N, V), 0.0), 4.0);
        col = mix(col, vec3(0.85, 0.93, 1.0), 0.45 * fres);

        if (uShowHTilde == 1) {
            float t = clamp(vHTilde * uHTildeGain, -1.0, 1.0);
            vec3  warm = vec3(1.00, 0.55, 0.35);
            vec3  cool = vec3(0.20, 0.55, 1.00);
            vec3  tint = t >= 0.0 ? warm : cool;
            col = mix(col, tint, min(0.6, abs(t)));
        }

        // Soft fade-in over the first centimetre of water.
        float a = uBaseColor.a * smoothstep(0.005, 0.05, vH);
        FragColor = vec4(col, a);
    } else {
        FragColor = vec4(col, 1.0);
    }
}
)GLSL";

// ---------------------------------------------------------------------------
// GL helpers.
// ---------------------------------------------------------------------------
static GLuint CompileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048]; GLsizei n = 0;
        glGetShaderInfoLog(s, sizeof(log), &n, log);
        std::fprintf(stderr, "[GL] shader compile error: %.*s\n", (int)n, log);
    }
    return s;
}
static GLuint MakeProgram(const char* vs, const char* fs) {
    GLuint v = CompileShader(GL_VERTEX_SHADER,   vs);
    GLuint f = CompileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048]; GLsizei n = 0;
        glGetProgramInfoLog(p, sizeof(log), &n, log);
        std::fprintf(stderr, "[GL] program link error: %.*s\n", (int)n, log);
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}
static void GLAPIENTRY GLDebugCB(GLenum, GLenum type, GLuint, GLenum severity,
                                 GLsizei, const GLchar* msg, const void*) {
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;
    std::fprintf(stderr, "[GL] type=0x%x sev=0x%x: %s\n", type, severity, msg);
}

// ---------------------------------------------------------------------------
// Build a static (i, j) grid mesh shared by terrain and water.
//   * One vec2 per vertex (i, j) in [0, N-1].
//   * Two triangles per cell.
// ---------------------------------------------------------------------------
struct GridMesh {
    GLuint vao = 0, vbo = 0, ibo = 0;
    GLsizei indexCount = 0;
};

static GridMesh BuildGridMesh(int N) {
    GridMesh m;
    std::vector<float> verts; verts.reserve((size_t)N * N * 2);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            verts.push_back((float)i);
            verts.push_back((float)j);
        }

    std::vector<unsigned int> idx; idx.reserve((size_t)(N - 1) * (N - 1) * 6);
    for (int j = 0; j < N - 1; ++j) {
        for (int i = 0; i < N - 1; ++i) {
            unsigned int v00 = (unsigned int)( j      * N + i    );
            unsigned int v10 = (unsigned int)( j      * N + i + 1);
            unsigned int v01 = (unsigned int)((j + 1) * N + i    );
            unsigned int v11 = (unsigned int)((j + 1) * N + i + 1);
            idx.push_back(v00); idx.push_back(v10); idx.push_back(v11);
            idx.push_back(v00); idx.push_back(v11); idx.push_back(v01);
        }
    }
    m.indexCount = (GLsizei)idx.size();

    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glGenBuffers(1, &m.ibo);
    glBindVertexArray(m.vao);

    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(unsigned int),
                 idx.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);
    return m;
}

static GLuint MakeFloatTexture(int N) {
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, N, N, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
    return t;
}

static void UploadFloatTexture(GLuint tex, int N, const float* data) {
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, N, N, GL_RED, GL_FLOAT, data);
}

// ---------------------------------------------------------------------------
// Camera state (orbit around grid centre).
// ---------------------------------------------------------------------------
struct OrbitCam {
    float yaw      = -0.8f;          // around +Y
    float pitch    =  0.55f;         // up from horizontal
    float dist     = 360.f;
    Vec3  target   = V3(0.f, 0.f, 0.f);

    Vec3 EyePos() const {
        float cp = std::cos(pitch), sp = std::sin(pitch);
        float cy = std::cos(yaw),   sy = std::sin(yaw);
        return target + V3(dist * cp * sy, dist * sp, dist * cp * cy);
    }
};

// ---------------------------------------------------------------------------
// Globals captured by GLFW callbacks.
// ---------------------------------------------------------------------------
struct App {
    int       fb_w = 1280, fb_h = 720;
    OrbitCam  cam;
    bool      mouseRotating = false;
    bool      mousePanning  = false;
    double    lastMx = 0, lastMy = 0;
};
static App g_app;

static void FramebufferSizeCB(GLFWwindow*, int w, int h) {
    g_app.fb_w = w;
    g_app.fb_h = h;
    glViewport(0, 0, w, h);
}
static void ScrollCB(GLFWwindow*, double, double yoff) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    g_app.cam.dist *= std::pow(0.9f, (float)yoff);
    g_app.cam.dist = std::clamp(g_app.cam.dist, 30.f, 1500.f);
}

// ---------------------------------------------------------------------------
// Ray-cast the cursor against the horizontal plane y = pickY in world space.
// Returns true and writes hitX/hitZ if there is a finite intersection in
// front of the camera.
// ---------------------------------------------------------------------------
static bool PickGroundXZ(const Mat4& proj, const Mat4& view, Vec3 eye,
                         double mx, double my, int fbW, int fbH,
                         float pickY, float& hitX, float& hitZ)
{
    // NDC: x in [-1, 1], y in [-1, 1] (note: GLFW Y-down -> NDC Y-up).
    float ndcX = (float)(2.0 * mx / std::max(1, fbW) - 1.0);
    float ndcY = (float)(1.0 - 2.0 * my / std::max(1, fbH));

    // Inverse(proj * view) applied to a far-plane NDC point.
    Mat4 PV = MatMul(proj, view);
    float* M = PV.m;
    // Compute inverse via cofactor expansion (general 4x4 inverse).
    float inv[16];
    inv[0]  =  M[5] * M[10] * M[15] - M[5] * M[11] * M[14] - M[9] * M[6] * M[15] + M[9] * M[7] * M[14] + M[13] * M[6] * M[11] - M[13] * M[7] * M[10];
    inv[4]  = -M[4] * M[10] * M[15] + M[4] * M[11] * M[14] + M[8] * M[6] * M[15] - M[8] * M[7] * M[14] - M[12] * M[6] * M[11] + M[12] * M[7] * M[10];
    inv[8]  =  M[4] * M[9]  * M[15] - M[4] * M[11] * M[13] - M[8] * M[5] * M[15] + M[8] * M[7] * M[13] + M[12] * M[5] * M[11] - M[12] * M[7] * M[9];
    inv[12] = -M[4] * M[9]  * M[14] + M[4] * M[10] * M[13] + M[8] * M[5] * M[14] - M[8] * M[6] * M[13] - M[12] * M[5] * M[10] + M[12] * M[6] * M[9];
    inv[1]  = -M[1] * M[10] * M[15] + M[1] * M[11] * M[14] + M[9] * M[2] * M[15] - M[9] * M[3] * M[14] - M[13] * M[2] * M[11] + M[13] * M[3] * M[10];
    inv[5]  =  M[0] * M[10] * M[15] - M[0] * M[11] * M[14] - M[8] * M[2] * M[15] + M[8] * M[3] * M[14] + M[12] * M[2] * M[11] - M[12] * M[3] * M[10];
    inv[9]  = -M[0] * M[9]  * M[15] + M[0] * M[11] * M[13] + M[8] * M[1] * M[15] - M[8] * M[3] * M[13] - M[12] * M[1] * M[11] + M[12] * M[3] * M[9];
    inv[13] =  M[0] * M[9]  * M[14] - M[0] * M[10] * M[13] - M[8] * M[1] * M[14] + M[8] * M[2] * M[13] + M[12] * M[1] * M[10] - M[12] * M[2] * M[9];
    inv[2]  =  M[1] * M[6]  * M[15] - M[1] * M[7]  * M[14] - M[5] * M[2] * M[15] + M[5] * M[3] * M[14] + M[13] * M[2] * M[7]  - M[13] * M[3] * M[6];
    inv[6]  = -M[0] * M[6]  * M[15] + M[0] * M[7]  * M[14] + M[4] * M[2] * M[15] - M[4] * M[3] * M[14] - M[12] * M[2] * M[7]  + M[12] * M[3] * M[6];
    inv[10] =  M[0] * M[5]  * M[15] - M[0] * M[7]  * M[13] - M[4] * M[1] * M[15] + M[4] * M[3] * M[13] + M[12] * M[1] * M[7]  - M[12] * M[3] * M[5];
    inv[14] = -M[0] * M[5]  * M[14] + M[0] * M[6]  * M[13] + M[4] * M[1] * M[14] - M[4] * M[2] * M[13] - M[12] * M[1] * M[6]  + M[12] * M[2] * M[5];
    inv[3]  = -M[1] * M[6]  * M[11] + M[1] * M[7]  * M[10] + M[5] * M[2] * M[11] - M[5] * M[3] * M[10] - M[9]  * M[2] * M[7]  + M[9]  * M[3] * M[6];
    inv[7]  =  M[0] * M[6]  * M[11] - M[0] * M[7]  * M[10] - M[4] * M[2] * M[11] + M[4] * M[3] * M[10] + M[8]  * M[2] * M[7]  - M[8]  * M[3] * M[6];
    inv[11] = -M[0] * M[5]  * M[11] + M[0] * M[7]  * M[9]  + M[4] * M[1] * M[11] - M[4] * M[3] * M[9]  - M[8]  * M[1] * M[7]  + M[8]  * M[3] * M[5];
    inv[15] =  M[0] * M[5]  * M[10] - M[0] * M[6]  * M[9]  - M[4] * M[1] * M[10] + M[4] * M[2] * M[9]  + M[8]  * M[1] * M[6]  - M[8]  * M[2] * M[5];
    float det = M[0] * inv[0] + M[1] * inv[4] + M[2] * inv[8] + M[3] * inv[12];
    if (std::abs(det) < 1e-20f) return false;
    float invDet = 1.f / det;
    for (int k = 0; k < 16; ++k) inv[k] *= invDet;

    auto unproj = [&](float nx, float ny, float nz) {
        float x = inv[0] * nx + inv[4] * ny + inv[8]  * nz + inv[12];
        float y = inv[1] * nx + inv[5] * ny + inv[9]  * nz + inv[13];
        float z = inv[2] * nx + inv[6] * ny + inv[10] * nz + inv[14];
        float w = inv[3] * nx + inv[7] * ny + inv[11] * nz + inv[15];
        if (std::abs(w) < 1e-20f) w = 1e-20f;
        return V3(x / w, y / w, z / w);
    };
    Vec3 nearP = unproj(ndcX, ndcY, -1.f);
    Vec3 farP  = unproj(ndcX, ndcY,  1.f);
    Vec3 dir = Normalize(farP - nearP);

    if (std::abs(dir.y) < 1e-6f) return false;
    float t = (pickY - eye.y) / dir.y;
    if (t <= 0.f) return false;
    Vec3 hit = eye + dir * t;
    hitX = hit.x;
    hitZ = hit.z;
    return true;
}

// ===========================================================================
// main
// ===========================================================================
int main(int /*argc*/, char** /*argv*/)
{
    glfwSetErrorCallback([](int code, const char* desc) {
        std::fprintf(stderr, "[GLFW] error %d: %s\n", code, desc);
    });
    if (!glfwInit()) {
        std::fprintf(stderr, "glfwInit failed\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,        GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#ifndef NDEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT,  GLFW_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Dispersive SWE 2D (CUDA)", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetFramebufferSizeCallback(window, FramebufferSizeCB);
    glfwSetScrollCallback(window, ScrollCB);
    glfwGetFramebufferSize(window, &g_app.fb_w, &g_app.fb_h);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::fprintf(stderr, "gladLoadGL failed\n");
        return 1;
    }
#ifndef NDEBUG
    if (GLAD_GL_KHR_debug) {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(GLDebugCB, nullptr);
    }
#endif
    std::printf("OpenGL %s\nGLSL %s\nGPU %s\n",
                glGetString(GL_VERSION),
                glGetString(GL_SHADING_LANGUAGE_VERSION),
                glGetString(GL_RENDERER));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    // ---------- Simulation ----------
    // Sim2D holds ~2.5 MB of host shadow arrays as members; the default
    // Windows stack (1 MB) cannot fit it, so always heap-allocate.
    auto simPtr = std::make_unique<Sim2D>();
    Sim2D& sim  = *simPtr;
    sim.SyncToHost();
    const int   N      = GRIDRESOLUTION_2D;
    const float center = 0.5f * (float)N;
    g_app.cam.target   = V3(center, 0.f, center);

    // ---------- GL resources ----------
    GLuint prog          = MakeProgram(kVertSrc, kFragSrc);
    GLint  locVP         = glGetUniformLocation(prog, "uVP");
    GLint  locTerrain    = glGetUniformLocation(prog, "uTerrain");
    GLint  locWaterH     = glGetUniformLocation(prog, "uWaterH");
    GLint  locHTilde     = glGetUniformLocation(prog, "uHTilde");
    GLint  locTexel      = glGetUniformLocation(prog, "uTexel");
    GLint  locYScale     = glGetUniformLocation(prog, "uYScale");
    GLint  locIsWater    = glGetUniformLocation(prog, "uIsWater");
    GLint  locLightDir   = glGetUniformLocation(prog, "uLightDir");
    GLint  locViewPos    = glGetUniformLocation(prog, "uViewPos");
    GLint  locBaseColor  = glGetUniformLocation(prog, "uBaseColor");
    GLint  locHTildeGain = glGetUniformLocation(prog, "uHTildeGain");
    GLint  locShowHT     = glGetUniformLocation(prog, "uShowHTilde");

    GridMesh mesh        = BuildGridMesh(N);
    GLuint   terrainTex  = MakeFloatTexture(N);
    GLuint   hTex        = MakeFloatTexture(N);
    GLuint   htildeTex   = MakeFloatTexture(N);

    UploadFloatTexture(terrainTex, N, sim.terrain);
    UploadFloatTexture(hTex,       N, sim.h);
    UploadFloatTexture(htildeTex,  N, sim.htilde);

    // ---------- UI / interaction state ----------
    int   resetTerrainType = 0;
    int   resetWaterType   = 0;
    float resetWaterLevel  = 12.0f;
    bool  sweOnly          = false;
    bool  paused           = false;
    bool  stepOnce         = false;
    int   substeps         = 1;
    int   slowMo           = 2;          // run one sim step every slowMo frames (1 = realtime)
    int   slowMoAcc        = 0;

    float editSize         = 0.06f;
    float editFactor       = 1.0f;

    bool  showTerrain      = true;
    bool  showWater        = true;
    bool  wireWater        = false;
    bool  showHTildeOverlay = true;
    float htildeGain       = 6.0f;
    float yScale           = 1.0f;
    bool  showGrid         = true;     // (visual: a faint frame around domain)

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // --------- Step the simulation ---------
        if (!paused) {
            // slowMo > 1  =>  only step once every slowMo frames
            if (++slowMoAcc >= slowMo) {
                slowMoAcc = 0;
                for (int s = 0; s < substeps; ++s) sim.SimStep(sweOnly);
            }
        } else if (stepOnce) {
            sim.SimStep(sweOnly);
            stepOnce = false;
            slowMoAcc = 0;
        }

        // --------- Refresh dynamic textures ---------
        UploadFloatTexture(hTex,      N, sim.h);
        UploadFloatTexture(htildeTex, N, sim.htilde);

        // --------- ImGui frame ---------
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(360, 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("Dispersive SWE 2D");

        ImGui::Text("Sim time : %7.2f s", sim.time);
        ImGui::Text("Frame    : %5.1f FPS  (%5.2f ms)",
                    io.Framerate, 1000.f / std::max(1.f, io.Framerate));
        ImGui::Text("Resolution: N = %d   dx = %d   dt = %.4f",
                    N, GRIDCELLSIZE_2D, (float)TIMESTEP_2D);

        ImGui::Separator();
        ImGui::TextUnformatted("Terrain");
        ImGui::RadioButton("Flat",  &resetTerrainType, 0); ImGui::SameLine();
        ImGui::RadioButton("Hills", &resetTerrainType, 1);
        if (ImGui::Button("Reset terrain")) {
            sim.ResetTerrain(resetTerrainType);
            sim.ResetWater(resetWaterType, resetWaterLevel);
            sim.SyncToHost();
            UploadFloatTexture(terrainTex, N, sim.terrain);
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Initial water");
        ImGui::RadioButton("Const",     &resetWaterType, 0); ImGui::SameLine();
        ImGui::RadioButton("Dam break", &resetWaterType, 1);
        ImGui::RadioButton("Sloped",    &resetWaterType, 2); ImGui::SameLine();
        ImGui::RadioButton("Cosine",    &resetWaterType, 3);
        ImGui::SliderFloat("Level (y)", &resetWaterLevel, -8.f, 12.f, "%.2f");
        if (ImGui::Button("Reset water")) {
            sim.ResetWater(resetWaterType, resetWaterLevel);
            sim.SyncToHost();
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Solver");
        ImGui::Checkbox("SWE only (no Airy waves)", &sweOnly);
        ImGui::Checkbox("Pause", &paused); ImGui::SameLine();
        if (ImGui::Button("Step")) stepOnce = true;
        ImGui::SliderInt("Substeps / frame", &substeps, 1, 8);
        ImGui::SliderInt("Slow-mo (1 step / N frames)", &slowMo, 1, 16);

        ImGui::Separator();
        ImGui::TextUnformatted("Boundary");
        if (ImGui::RadioButton("Reflective walls", &sim.boundaryMode, 0)) {
            sim.ResetTerrain(resetTerrainType);
            sim.ResetWater(resetWaterType, resetWaterLevel);
            sim.SyncToHost();
            UploadFloatTexture(terrainTex, N, sim.terrain);
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Absorbing sponge", &sim.boundaryMode, 1)) {
            sim.ResetTerrain(resetTerrainType);
            sim.ResetWater(resetWaterType, resetWaterLevel);
            sim.SyncToHost();
            UploadFloatTexture(terrainTex, N, sim.terrain);
        }
        if (sim.boundaryMode == 1) {
            sim.spongeWidth    = 1;
            sim.spongeStrength = 0.f;
            ImGui::Text("Sponge width:    %d (fixed)", sim.spongeWidth);
            ImGui::Text("Sponge strength: %.3f (fixed)", sim.spongeStrength);
            ImGui::Text("Rest water level: %.2f (set by Reset water)", sim.restWaterLevel);
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Brush  (MMB = add, +Shift = remove)");
        ImGui::SliderFloat("Brush size",   &editSize,   0.01f, 0.30f, "%.3f");
        ImGui::SliderFloat("Brush factor", &editFactor, 0.f,   8.f,   "%.2f");

        ImGui::Separator();
        ImGui::TextUnformatted("View");
        ImGui::Checkbox("Terrain", &showTerrain); ImGui::SameLine();
        ImGui::Checkbox("Water",   &showWater);
        ImGui::Checkbox("Wireframe water", &wireWater);
        ImGui::Checkbox("htilde colour overlay", &showHTildeOverlay);
        ImGui::SliderFloat("htilde gain",  &htildeGain, 0.5f, 50.f, "%.1fx");
        ImGui::SliderFloat("Y exaggerate", &yScale,     0.5f,  6.f, "%.2fx");
        ImGui::Checkbox("Bounding frame",  &showGrid);

        ImGui::TextDisabled("LMB-drag: orbit | RMB-drag: pan | Wheel: zoom");
        ImGui::End();

        // --------- Camera input ---------
        bool overUI = io.WantCaptureMouse;
        double mx, my; glfwGetCursorPos(window, &mx, &my);
        bool lmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)   == GLFW_PRESS;
        bool rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)  == GLFW_PRESS;
        bool mmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;

        if (!overUI && lmb && !g_app.mouseRotating) {
            g_app.mouseRotating = true;
            g_app.lastMx = mx; g_app.lastMy = my;
        } else if (!lmb) g_app.mouseRotating = false;
        if (!overUI && rmb && !g_app.mousePanning) {
            g_app.mousePanning = true;
            g_app.lastMx = mx; g_app.lastMy = my;
        } else if (!rmb) g_app.mousePanning = false;

        if (g_app.mouseRotating) {
            float dx = (float)(mx - g_app.lastMx);
            float dy = (float)(my - g_app.lastMy);
            g_app.cam.yaw   -= dx * 0.005f;
            g_app.cam.pitch += dy * 0.005f;
            g_app.cam.pitch = std::clamp(g_app.cam.pitch, -1.5f, 1.5f);
            g_app.lastMx = mx; g_app.lastMy = my;
        }
        if (g_app.mousePanning) {
            float dx = (float)(mx - g_app.lastMx);
            float dy = (float)(my - g_app.lastMy);
            float panScale = g_app.cam.dist * 0.0015f;
            // Pan in the camera-aligned screen plane (xz only for stability).
            float cy = std::cos(g_app.cam.yaw),  sy = std::sin(g_app.cam.yaw);
            Vec3 right   = V3( cy, 0.f, -sy);
            Vec3 forward = V3( sy, 0.f,  cy);
            g_app.cam.target = g_app.cam.target
                             - right   * (dx * panScale)
                             + forward * (dy * panScale);
            g_app.lastMx = mx; g_app.lastMy = my;
        }

        // --------- Build matrices ---------
        float aspect = (float)g_app.fb_w / std::max(1, g_app.fb_h);
        Mat4 proj = MatPerspective(60.f * 3.14159265f / 180.f, aspect, 1.f, 4000.f);
        Vec3 eye  = g_app.cam.EyePos();
        Mat4 view = MatLookAt(eye, g_app.cam.target, V3(0.f, 1.f, 0.f));
        Mat4 vp   = MatMul(proj, view);

        // --------- Brush via middle-click (or shift-MMB to dig) ---------
        if (!overUI && mmb) {
            float pickY = resetWaterLevel * yScale;
            float hx, hz;
            if (PickGroundXZ(proj, view, eye, mx, my, g_app.fb_w, g_app.fb_h,
                             pickY, hx, hz)) {
                float xN = std::clamp(hx / (float)N, 0.f, 1.f);
                float yN = std::clamp(hz / (float)N, 0.f, 1.f);
                bool subtract = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)  == GLFW_PRESS) ||
                                (glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
                float sgn = subtract ? -1.f : 1.f;
                sim.EditWaterLocal(xN, yN, editSize, sgn * editFactor * (float)TIMESTEP_2D);
                sim.SyncToHost();
                UploadFloatTexture(hTex, N, sim.h);
            }
        }

        // --------- Render ---------
        glViewport(0, 0, g_app.fb_w, g_app.fb_h);
        glClearColor(0.07f, 0.09f, 0.13f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        glUseProgram(prog);
        glUniformMatrix4fv(locVP, 1, GL_FALSE, vp.m);
        glUniform1f(locTexel, 1.f / (float)N);
        glUniform1f(locYScale, yScale);
        glUniform3f(locLightDir, -0.45f, -0.85f, -0.30f);
        glUniform3f(locViewPos, eye.x, eye.y, eye.z);
        glUniform1f(locHTildeGain, htildeGain);
        glUniform1i(locShowHT, showHTildeOverlay ? 1 : 0);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, terrainTex); glUniform1i(locTerrain, 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, hTex);       glUniform1i(locWaterH,  1);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, htildeTex);  glUniform1i(locHTilde,  2);

        glBindVertexArray(mesh.vao);

        // Terrain pass.
        if (showTerrain) {
            glDisable(GL_BLEND);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glUniform1i(locIsWater, 0);
            glUniform4f(locBaseColor, 0.45f, 0.32f, 0.20f, 1.f);
            glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, 0);
        }

        // Water pass (transparent, no back-face culling so we still see the
        // underside through translucency).
        if (showWater) {
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDisable(GL_CULL_FACE);
            glDepthMask(GL_FALSE);

            glPolygonMode(GL_FRONT_AND_BACK, wireWater ? GL_LINE : GL_FILL);

            glUniform1i(locIsWater, 1);
            glUniform4f(locBaseColor, 0.30f, 0.55f, 0.85f, wireWater ? 0.9f : 0.65f);
            glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, 0);

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glDepthMask(GL_TRUE);
            glEnable(GL_CULL_FACE);
        }

        // Optional bounding frame for spatial reference.
        if (showGrid) {
            // Drawing a plain colour box reuses the same shader (uIsWater = 0
            // with terrain disabled by binding a flat texture would be more
            // work than this is worth -- skip and let the user infer scale
            // from the terrain). Keep the toggle for future use.
        }

        glBindVertexArray(0);

        // --------- ImGui draw ---------
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // --------- Shutdown ---------
    sim.Release();
    glDeleteTextures(1, &terrainTex);
    glDeleteTextures(1, &hTex);
    glDeleteTextures(1, &htildeTex);
    glDeleteBuffers(1, &mesh.vbo);
    glDeleteBuffers(1, &mesh.ibo);
    glDeleteVertexArrays(1, &mesh.vao);
    glDeleteProgram(prog);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
