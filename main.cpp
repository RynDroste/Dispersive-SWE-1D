// ============================================================================
// main.cpp -- Modern OpenGL 3.3 + Dear ImGui frontend for the 1D dispersive
// shallow-water simulator (Sim.cu / Sim.h).
//
// What it draws (left-to-right cross section, x in [0, GRIDRESOLUTION)):
//   * Terrain                     : filled brown polygon
//   * Total water (terrain + h)   : filled blue polygon + bright surface line
//   * Bulk surface (hbar + terr.) : amber line                        (toggle)
//   * Surface displacement htilde : red overlay near the bottom        (toggle)
//   * Flow rate q                 : faint cyan line near the bottom    (toggle)
//
// What you can do via ImGui:
//   * Reset terrain (flat / hill)
//   * Reset water (constant / dam-break / sloped / cosine wave) + level slider
//   * Toggle SWE-only / Pause / Single-step
//   * Left-click on the canvas to ADD water locally (right-click to REMOVE)
//   * Sliders for brush size and brush strength
//   * Toggle which overlays are visible
//
// Build dependencies (provide via your project / vcpkg / manual):
//   * GLFW 3.x          ( <GLFW/glfw3.h>, glfw3.lib )
//   * glad (GL 3.3 Core) ( <glad/glad.h>, glad.c )
//   * Dear ImGui sources +
//       backends/imgui_impl_glfw.cpp,  backends/imgui_impl_opengl3.cpp
//   * CUDA Toolkit (for Sim.cu --> cudart.lib, cufft.lib)
//
// Example one-shot build (Visual Studio Developer PowerShell):
//   nvcc -O2 -std=c++17 ^
//        Sim.cu main.cpp glad.c ^
//        imgui.cpp imgui_draw.cpp imgui_widgets.cpp imgui_tables.cpp ^
//        backends/imgui_impl_glfw.cpp backends/imgui_impl_opengl3.cpp ^
//        -I<imgui_dir> -I<imgui_dir>/backends ^
//        -I<glfw_inc> -I<glad_inc> ^
//        -L<glfw_lib> -lglfw3 -lcufft ^
//        -lopengl32 -lgdi32 -luser32 -lshell32 ^
//        -o sim_viewer.exe
// ============================================================================

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "Sim.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Embedded GLSL shaders (color-only, fed by a single ortho projection matrix).
// ---------------------------------------------------------------------------
static const char* kVertSrc = R"GLSL(
#version 330 core
layout(location = 0) in vec2 aPos;
uniform mat4 uProj;
void main() { gl_Position = uProj * vec4(aPos, 0.0, 1.0); }
)GLSL";

static const char* kFragSrc = R"GLSL(
#version 330 core
uniform vec4 uColor;
out vec4 FragColor;
void main() { FragColor = uColor; }
)GLSL";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void OrthoMat(float* m, float l, float r, float b, float t, float n, float f)
{
    std::memset(m, 0, 16 * sizeof(float));
    m[0]  =  2.f / (r - l);
    m[5]  =  2.f / (t - b);
    m[10] = -2.f / (f - n);
    m[12] = -(r + l) / (r - l);
    m[13] = -(t + b) / (t - b);
    m[14] = -(f + n) / (f - n);
    m[15] =  1.f;
}

static GLuint CompileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024]; GLsizei n = 0;
        glGetShaderInfoLog(s, sizeof(log), &n, log);
        std::fprintf(stderr, "[GL] shader compile error: %.*s\n", (int)n, log);
    }
    return s;
}

static GLuint MakeProgram()
{
    GLuint vs = CompileShader(GL_VERTEX_SHADER,   kVertSrc);
    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, kFragSrc);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024]; GLsizei n = 0;
        glGetProgramInfoLog(p, sizeof(log), &n, log);
        std::fprintf(stderr, "[GL] program link error: %.*s\n", (int)n, log);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

static void GLAPIENTRY GLDebugCallback(GLenum, GLenum type, GLuint, GLenum severity,
                                       GLsizei, const GLchar* msg, const void*)
{
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;
    std::fprintf(stderr, "[GL] type=0x%x sev=0x%x: %s\n", type, severity, msg);
}

// ---------------------------------------------------------------------------
// State for the viewer (kept here so callbacks can find it)
// ---------------------------------------------------------------------------
struct View {
    int   fb_w = 1280, fb_h = 720;
    float ymin = -25.f, ymax = 25.f;
};
static View g_view;

static void FramebufferSizeCB(GLFWwindow* /*win*/, int w, int h)
{
    g_view.fb_w = w;
    g_view.fb_h = h;
    glViewport(0, 0, w, h);
}

// Convert a screen-space pixel position to the world x in [0, GRIDRESOLUTION).
static float ScreenXToWorldX(double px)
{
    float t = (float)(px / std::max(1, g_view.fb_w));
    return t * (float)GRIDRESOLUTION;
}

// ---------------------------------------------------------------------------
// Vertex helpers (pure host-side line/strip building).
// ---------------------------------------------------------------------------
static inline void PushXY(std::vector<float>& v, float x, float y)
{
    v.push_back(x);
    v.push_back(y);
}

static void DrawArray(GLuint vbo, const std::vector<float>& v, GLenum mode,
                      GLint locColor, float r, float g, float b, float a,
                      float lineWidth = 1.f)
{
    if (v.empty()) return;
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, v.size() * sizeof(float), v.data());
    glUniform4f(locColor, r, g, b, a);
    if (mode == GL_LINE_STRIP || mode == GL_LINES) glLineWidth(lineWidth);
    glDrawArrays(mode, 0, (GLsizei)(v.size() / 2));
}

// ===========================================================================
//  main
// ===========================================================================
int main(int /*argc*/, char** /*argv*/)
{
    // ----- GLFW window + OpenGL 3.3 Core context -----
    glfwSetErrorCallback([](int code, const char* desc) {
        std::fprintf(stderr, "[GLFW] error %d: %s\n", code, desc);
    });
    if (!glfwInit()) {
        std::fprintf(stderr, "glfwInit failed\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#ifndef NDEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Dispersive SWE 1D (CUDA)", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetFramebufferSizeCallback(window, FramebufferSizeCB);
    glfwGetFramebufferSize(window, &g_view.fb_w, &g_view.fb_h);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::fprintf(stderr, "gladLoadGL failed\n");
        return 1;
    }
#ifndef NDEBUG
    if (GLAD_GL_KHR_debug) {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(GLDebugCallback, nullptr);
    }
#endif
    std::printf("OpenGL %s\nGLSL %s\nGPU %s\n",
                glGetString(GL_VERSION),
                glGetString(GL_SHADING_LANGUAGE_VERSION),
                glGetString(GL_RENDERER));

    // ----- Dear ImGui -----
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    // ----- Simulation -----
    Sim sim;
    sim.SyncToHost();   // populate host shadows for the very first frame

    // ----- GL resources -----
    GLuint prog     = MakeProgram();
    GLint  locProj  = glGetUniformLocation(prog, "uProj");
    GLint  locColor = glGetUniformLocation(prog, "uColor");

    GLuint vao = 0, vbo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // Reserve enough for the largest single draw (terrain triangle strip = 2*N verts).
    const GLsizeiptr kVboBytes = 4 * GRIDRESOLUTION * (GLsizeiptr)sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, kVboBytes, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glBindVertexArray(0);

    std::vector<float> verts;
    verts.reserve(4 * GRIDRESOLUTION);

    // ----- UI / interaction state -----
    int   resetTerrainType = 1;          // 0=flat, 1=hill (for one-shot reset buttons)
    int   resetWaterType   = 2;          // 0=const, 1=dam, 2=sloped, 3=cosine
    float resetWaterLevel  = 0.f;
    bool  sweOnly          = false;
    bool  paused           = false;
    bool  stepOnce         = false;
    int   substeps         = 1;          // simulator sub-steps per rendered frame

    float editSize         = 0.04f;      // brush radius in normalized x
    float editFactor       = 1.0f;       // brush strength multiplier (per click frame)

    bool  showTerrain      = true;
    bool  showWaterFill    = true;
    bool  showWaterLine    = true;
    bool  showHbar         = false;
    bool  showHtilde       = true;
    bool  showQ            = false;
    float overlayScale     = 5.f;        // visual amplification for tiny htilde / q

    bool  autoFitY         = true;       // auto-adjust vertical extent each frame

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // ============== Step the simulation ==============
        if (!paused) {
            for (int s = 0; s < substeps; ++s) sim.SimStep(sweOnly);
            // SimStep already calls SyncToHost() at its tail.
        } else if (stepOnce) {
            sim.SimStep(sweOnly);
            stepOnce = false;
        }

        // ============== Build ImGui UI ==============
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(360, 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("Dispersive SWE 1D");

        ImGui::Text("Sim time : %7.2f s", sim.time);
        ImGui::Text("Frame    : %5.1f FPS  (%5.2f ms)",
                    io.Framerate, 1000.f / std::max(1.f, io.Framerate));
        ImGui::Text("Resolution: N = %d   dx = %d   dt = %.4f",
                    GRIDRESOLUTION, GRIDCELLSIZE, (float)TIMESTEP);

        ImGui::Separator();
        ImGui::TextUnformatted("Terrain");
        ImGui::RadioButton("Flat",  &resetTerrainType, 0); ImGui::SameLine();
        ImGui::RadioButton("Hill",  &resetTerrainType, 1);
        if (ImGui::Button("Reset terrain")) {
            sim.ResetTerrain(resetTerrainType);
            sim.ResetWater(resetWaterType, resetWaterLevel);
            sim.SyncToHost();
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Initial water");
        ImGui::RadioButton("Const",     &resetWaterType, 0); ImGui::SameLine();
        ImGui::RadioButton("Dam break", &resetWaterType, 1); ImGui::SameLine();
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
        ImGui::Checkbox("Pause", &paused);
        ImGui::SameLine();
        if (ImGui::Button("Step")) stepOnce = true;
        ImGui::SliderInt("Substeps / frame", &substeps, 1, 8);

        ImGui::Separator();
        ImGui::TextUnformatted("Brush  (left = add, right = remove)");
        ImGui::SliderFloat("Brush size",   &editSize,   0.005f, 0.25f, "%.3f");
        ImGui::SliderFloat("Brush factor", &editFactor, 0.f,    4.f,   "%.2f");

        ImGui::Separator();
        ImGui::TextUnformatted("Overlays");
        ImGui::Checkbox("Terrain",                   &showTerrain);
        ImGui::Checkbox("Water (fill)",              &showWaterFill);
        ImGui::Checkbox("Water surface line",        &showWaterLine);
        ImGui::Checkbox("Bulk surface (hbar + terr)", &showHbar);
        ImGui::Checkbox("Surface displacement (htilde)", &showHtilde);
        ImGui::Checkbox("Flow rate (q)",             &showQ);
        ImGui::SliderFloat("Overlay scale", &overlayScale, 0.5f, 50.f, "%.1fx");
        ImGui::Checkbox("Auto-fit Y range", &autoFitY);
        if (!autoFitY) {
            ImGui::SliderFloat("Y min", &g_view.ymin, -50.f,  10.f, "%.1f");
            ImGui::SliderFloat("Y max", &g_view.ymax, -10.f,  50.f, "%.1f");
        }
        ImGui::End();

        // ----- Mouse hover info window -----
        if (!io.WantCaptureMouse) {
            float wx = ScreenXToWorldX(io.MousePos.x);
            int   ix = std::clamp((int)std::floor(wx), 0, GRIDRESOLUTION - 1);
            ImGui::SetNextWindowPos(io.MousePos, 0, ImVec2(0.f, 1.2f));
            ImGui::SetNextWindowBgAlpha(0.6f);
            ImGui::Begin("hover", nullptr,
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
                ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove);
            ImGui::Text("x = %3d  terrain = %6.2f", ix, sim.terrain[ix]);
            ImGui::Text("h    = %6.3f   q    = %+7.3f", sim.h[ix],    sim.q[ix]);
            ImGui::Text("hbar = %6.3f  qbar = %+7.3f", sim.hbar[ix], sim.qbar[ix]);
            ImGui::Text("htil = %+6.3f qtil = %+7.3f", sim.htilde[ix], sim.qtilde[ix]);
            ImGui::End();
        }

        // ----- Mouse interaction (skip if ImGui owns the mouse) -----
        if (!io.WantCaptureMouse) {
            float xN = (float)(io.MousePos.x / std::max(1, g_view.fb_w));
            xN = std::clamp(xN, 0.f, 1.f);
            if (io.MouseDown[0]) {
                sim.EditWaterLocal(xN, editSize,  editFactor * (float)TIMESTEP);
                sim.SyncToHost();
            } else if (io.MouseDown[1]) {
                sim.EditWaterLocal(xN, editSize, -editFactor * (float)TIMESTEP);
                sim.SyncToHost();
            }
        }

        // ============== Compute world-space Y bounds ==============
        if (autoFitY) {
            float lo =  1e30f, hi = -1e30f;
            for (int x = 0; x < GRIDRESOLUTION; ++x) {
                lo = std::min(lo, sim.terrain[x]);
                hi = std::max(hi, sim.terrain[x] + sim.h[x]);
            }
            // a little margin + smooth EMA so the camera doesn't twitch
            float target_lo = lo - 2.f;
            float target_hi = hi + 3.f;
            const float a = 0.06f;
            g_view.ymin = (1.f - a) * g_view.ymin + a * target_lo;
            g_view.ymax = (1.f - a) * g_view.ymax + a * target_hi;
            if (g_view.ymax - g_view.ymin < 4.f) g_view.ymax = g_view.ymin + 4.f;
        }

        // ============== Render OpenGL scene ==============
        glViewport(0, 0, g_view.fb_w, g_view.fb_h);
        glClearColor(0.07f, 0.09f, 0.13f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        float proj[16];
        OrthoMat(proj, 0.f, (float)GRIDRESOLUTION, g_view.ymin, g_view.ymax, -1.f, 1.f);

        glUseProgram(prog);
        glUniformMatrix4fv(locProj, 1, GL_FALSE, proj);
        glBindVertexArray(vao);

        // -- Terrain (filled triangle strip from y_min up to terrain[x]) --
        if (showTerrain) {
            verts.clear();
            for (int x = 0; x < GRIDRESOLUTION; ++x) {
                PushXY(verts, (float)x, g_view.ymin);
                PushXY(verts, (float)x, sim.terrain[x]);
            }
            DrawArray(vbo, verts, GL_TRIANGLE_STRIP, locColor,
                      0.45f, 0.32f, 0.20f, 1.f);
        }

        // -- Water fill (semi-transparent triangle strip) --
        if (showWaterFill) {
            verts.clear();
            for (int x = 0; x < GRIDRESOLUTION; ++x) {
                float top = sim.terrain[x] + sim.h[x];
                PushXY(verts, (float)x, sim.terrain[x]);
                PushXY(verts, (float)x, top);
            }
            DrawArray(vbo, verts, GL_TRIANGLE_STRIP, locColor,
                      0.30f, 0.55f, 0.85f, 0.55f);
        }

        // -- Water surface line --
        if (showWaterLine) {
            verts.clear();
            for (int x = 0; x < GRIDRESOLUTION; ++x)
                PushXY(verts, (float)x, sim.terrain[x] + sim.h[x]);
            DrawArray(vbo, verts, GL_LINE_STRIP, locColor,
                      0.85f, 0.95f, 1.0f, 1.f, /*lineWidth=*/2.f);
        }

        // -- Bulk surface (hbar + terrain) overlay --
        if (showHbar) {
            verts.clear();
            for (int x = 0; x < GRIDRESOLUTION; ++x)
                PushXY(verts, (float)x, sim.terrain[x] + sim.hbar[x]);
            DrawArray(vbo, verts, GL_LINE_STRIP, locColor,
                      1.f, 0.7f, 0.2f, 0.9f, /*lineWidth=*/1.6f);
        }

        // -- Surface displacement htilde overlay (drawn at the bottom band) --
        if (showHtilde) {
            float baseY = g_view.ymin + 1.0f;
            // baseline marker
            verts.clear();
            PushXY(verts, 0.f,                       baseY);
            PushXY(verts, (float)GRIDRESOLUTION,     baseY);
            DrawArray(vbo, verts, GL_LINES, locColor, 0.5f, 0.5f, 0.55f, 0.6f, 1.f);

            verts.clear();
            for (int x = 0; x < GRIDRESOLUTION; ++x)
                PushXY(verts, (float)x, baseY + overlayScale * sim.htilde[x]);
            DrawArray(vbo, verts, GL_LINE_STRIP, locColor,
                      1.0f, 0.4f, 0.4f, 0.95f, /*lineWidth=*/1.6f);
        }

        // -- Flow rate q overlay (also bottom band, slightly higher) --
        if (showQ) {
            float baseY = g_view.ymin + (showHtilde ? 2.5f : 1.0f);
            verts.clear();
            PushXY(verts, 0.f,                       baseY);
            PushXY(verts, (float)GRIDRESOLUTION,     baseY);
            DrawArray(vbo, verts, GL_LINES, locColor, 0.5f, 0.5f, 0.55f, 0.6f, 1.f);

            verts.clear();
            for (int x = 0; x < GRIDRESOLUTION; ++x)
                PushXY(verts, (float)x, baseY + overlayScale * sim.q[x]);
            DrawArray(vbo, verts, GL_LINE_STRIP, locColor,
                      0.4f, 1.0f, 1.0f, 0.95f, /*lineWidth=*/1.4f);
        }

        glBindVertexArray(0);

        // ============== ImGui draw on top ==============
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // ----- Shutdown -----
    sim.Release();

    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
