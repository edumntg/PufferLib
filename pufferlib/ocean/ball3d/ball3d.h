#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include <math.h>

#define MAX_MOVEMENTS 1024
#define WIDTH 1080
#define HEIGHT 720
#define MIN_GOAL_POS -10.0f
#define MAX_GOAL_POS 10.0f

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};

typedef struct Client Client;
struct Client {
    Camera3D camera;
    float width;
    float height;

    float camera_distance;
    float camera_azimuth;
    float camera_elevation;
    bool is_dragging;
    Vector2 last_mouse_pos;
};

// Only use floats!
typedef struct {
    float score;
    float n; // Required as the last field
    float episode_length;
    float episode_return;
    float perf;
} Log;

typedef struct {
    float x;
    float y;
    float z;
} Vec3;

typedef struct {
    Log log;                     // Required field
    float* observations; // Required field. Ensure type matches in .py and .c
    float* actions;                // Required field. Ensure type matches in .py and .c
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field
    int size;

    Vec3 goal;
    Vec3 pos; // ball pos
    int ticks;

    Client* client;
} Ball3D;

// Add episode statistics to log
void add_log(Ball3D* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1.0f : 0.0f;
    env->log.score += env->rewards[0];
    env->log.episode_length += 1.0f;
    env->log.episode_return += env->rewards[0];
    env->log.n += 1.0f;
}

static inline float clampf(float v, float min, float max) {
    if (v < min)
        return min;
    if (v > max)
        return max;
    return v;
}


static inline float rndf(float a, float b) {
    return a + ((float)rand() / (float)RAND_MAX) * (b - a);
}

static inline Vec3 add3(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }

static inline Vec3 sub3(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }

static inline Vec3 scalmul3(Vec3 a, float b) { return (Vec3){a.x * b, a.y * b, a.z * b}; }

static inline float dot3(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static inline float norm3(Vec3 a) { return sqrtf(dot3(a, a)); }

static void update_camera_position(Client *c) {
    float r = c->camera_distance;
    float az = c->camera_azimuth;
    float el = c->camera_elevation;

    float x = r * cosf(el) * cosf(az);
    float y = r * cosf(el) * sinf(az);
    float z = r * sinf(el);

    c->camera.position = (Vector3){x, y, z};
    c->camera.target = (Vector3){0, 0, 0};
}

void handle_camera_controls(Client *client) {
    Vector2 mouse_pos = GetMousePosition();

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = true;
        client->last_mouse_pos = mouse_pos;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = false;
    }

    if (client->is_dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse_delta = {mouse_pos.x - client->last_mouse_pos.x,
                               mouse_pos.y - client->last_mouse_pos.y};

        float sensitivity = 0.005f;

        client->camera_azimuth -= mouse_delta.x * sensitivity;

        client->camera_elevation += mouse_delta.y * sensitivity;
        client->camera_elevation =
            clampf(client->camera_elevation, -PI / 2.0f + 0.1f, PI / 2.0f - 0.1f);

        client->last_mouse_pos = mouse_pos;

        update_camera_position(client);
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        client->camera_distance -= wheel * 2.0f;
        client->camera_distance = clampf(client->camera_distance, 5.0f, 50.0f);
        update_camera_position(client);
    }
}

void c_reset(Ball3D* env) {
    float dist = 25.0f;
    do {
        env->pos.x = rndf(MIN_GOAL_POS, MAX_GOAL_POS);
        env->pos.y = rndf(MIN_GOAL_POS, MAX_GOAL_POS);
        env->pos.z = rndf(MIN_GOAL_POS, MAX_GOAL_POS);

        env->goal.x = rndf(MIN_GOAL_POS, MAX_GOAL_POS);
        env->goal.y = rndf(MIN_GOAL_POS, MAX_GOAL_POS);
        env->goal.z = rndf(MIN_GOAL_POS, MAX_GOAL_POS);
        dist = norm3(sub3(env->pos, env->goal));
    } while(dist > 20.0f);

    // Initialize observations
    env->observations[0] = env->pos.x - env->goal.x;
    env->observations[1] = env->pos.y - env->goal.y;
    env->observations[2] = env->pos.z - env->goal.z;

    env->ticks = 0;

    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
}

void reset_ball(Ball3D* env) {
    env->pos.x = rndf(MIN_GOAL_POS, MAX_GOAL_POS);
    env->pos.y = rndf(MIN_GOAL_POS, MAX_GOAL_POS);
    env->pos.z = rndf(MIN_GOAL_POS, MAX_GOAL_POS);
}

void c_step(Ball3D* env) {
    env->ticks++;

    if(env->ticks >= MAX_MOVEMENTS) {
        c_reset(env);
        env->rewards[0] = -1.0f;
        env->terminals[0] = 1;
        env->log.n += 1.0f;
        return;
    }

    // Update ball positions
    env->pos.x += env->actions[0];
    env->pos.y += env->actions[1];
    env->pos.z += env->actions[2];

    // compute distance to goal
    float dist = norm3(sub3(env->pos, env->goal));
    if(dist > 20.0f) {
        reset_ball(env);
        env->rewards[0] = -1.0f;
        env->terminals[0] = 1;
        env->log.n += 1.0f;
    }

    if(dist < 1.0f) {
        reset_ball(env);
        env->rewards[0] = 1.0f;
        env->terminals[0] = 1;
        env->log.score += 1.0f;
        env->log.n += 1.0f;
    }

    // Update observations
    env->observations[0] = env->pos.x - env->goal.x;
    env->observations[1] = env->pos.y - env->goal.y;
    env->observations[2] = env->pos.z - env->goal.z;
}

void c_close(Ball3D* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

void c_render(Ball3D* env) {
    if (!IsWindowReady()) {
        Client *client = (Client *)calloc(1, sizeof(Client));

        client->width = WIDTH;
        client->height = HEIGHT;

        SetConfigFlags(FLAG_MSAA_4X_HINT); // antialiasing
        InitWindow(WIDTH, HEIGHT, "PufferLib DroneSwarm");

#ifndef __EMSCRIPTEN__
        SetTargetFPS(30);
#endif

        client->camera_distance = 40.0f;
        client->camera_azimuth = 0.0f;
        client->camera_elevation = PI / 10.0f;
        client->is_dragging = false;
        client->last_mouse_pos = (Vector2){0.0f, 0.0f};

        client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
        client->camera.fovy = 45.0f;
        client->camera.projection = CAMERA_PERSPECTIVE;
        env->client = client;

        update_camera_position(client);
    }

    if (WindowShouldClose()) {
        c_close(env);
        exit(0);
    }

    if (IsKeyDown(KEY_ESCAPE)) {

        c_close(env);
        exit(0);
    }

    handle_camera_controls(env->client);

    Client *client = env->client;

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    BeginMode3D(client->camera);
    DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, 20.0f, 20.0f, 20.0f, WHITE);

    DrawSphere((Vector3){env->pos.x, env->pos.y, env->pos.z}, 0.5f, PUFF_CYAN);

    DrawSphere((Vector3){env->goal.x, env->goal.y, env->goal.z}, 0.5f, PUFF_RED);
    EndMode3D();

    DrawText("Left click + drag: Rotate camera", 10, 10, 16, PUFF_WHITE);
    DrawText("Mouse wheel: Zoom in/out", 10, 30, 16, PUFF_WHITE);

    float dist = norm3(sub3(env->pos, env->goal));
    DrawText(TextFormat("Distance: %.4f", dist), 10, 50, 16, PUFF_WHITE);
    EndDrawing();
}
