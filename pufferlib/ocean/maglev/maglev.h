//
// Created by Eduardo Montilva on 20/7/25.
//

#include <math.h>
#include <stdbool.h>
#include <float.h>
#include "raylib.h"

#define G 9.81f // gravity constant
#define PI 3.14159265358979323846f // pi constant
#define DT 0.02f // time step for simulation
#define MAX_STEPS 200 // max sim steps (MAX_STEPS * DT = seconds)
#define eps 0.0001f // epsilon for floating point comparison
#define Y_MIN 0.0f
#define Y_MAX 2.0f
#define M 0.1f // mass in kg
#define K 0.05f // spring constant for magnetic force
#define X_TARGET 0.7f // desired position of ball in meters
#define V_MAX 10.0f // max velocity of ball in m/s
#define BALL_RADIUS 32 // Ball radius in pixels
#define WINDOW_WIDTH 400
#define WINDOW_HEIGHT 600
#define I_MAX 5.0f // max current

// Log struct for PufferLib
typedef struct {
    float perf; // 0-1 normalized performance metric
    float score; // unnormalized score
    float episode_return; // sum of agent rewards over episode
    float episode_length; // number of steps in episode
    float n; // Required as the last field
} Log;

// Environment struct
typedef struct {
    Log log; // Required field
    float* observations; // [position, velocity]
    float* actions; // [current]
    float* rewards; // [reward]
    unsigned char* terminals; // [done]
    unsigned char* truncations; // [truncated] (optional, but recommended)

    float x; // vertical position
    float v; // vertical velocity

    int tick; // step counter
    int size; // for window sizing (new)
} MagLev;

// Add episode statistics to log
void add_log(MagLev* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1.0f : 0.0f;
    env->log.score += env->rewards[0];
    env->log.episode_length += 1.0f;
    env->log.episode_return += env->rewards[0];
    env->log.n += 1.0f;
}

// Reset environment to initial state
void c_reset(MagLev* env) {
    env->x = Y_MIN + 0.1f; // start slightly above ground
    env->v = 0.0f;
    env->tick = 0;
    env->size = 1; // Default size for window scaling (can be set by binding)

    env->observations[0] = env->x; // position
    env->observations[1] = env->v; // velocity
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    if (env->truncations) env->truncations[0] = 0;
}

float clamp(float value, float min, float max) {
    return fminf(fmaxf(value, min), max);
}

// Step environment forward
void c_step(MagLev* env) {
    float a = env->actions[0];

    if (!isfinite(a)) {
        a = 0.0f;
    }
    a = clamp(a, -I_MAX, I_MAX); // Clamp current to max value
    env->actions[0] = a;

    // Magnetic force: F = k * i^2 / (x + eps)^2
    float balance = K * a * a / ((env->x + eps) * (env->x + eps)) - M * G;

    // Velocity change
    float v_dot = balance / M;
    env->v += DT * v_dot;

    // Position change
    float x_dot = env->v;
    env->x += DT * x_dot;
    if (env->x < Y_MIN) {
        env->x = Y_MIN;
        env->v = 0.0f;
    }

    env->tick += 1;

    // Termination and truncation
    bool terminated = env->x > Y_MAX || env->v < -V_MAX || env->v > V_MAX;
    bool truncated = env->tick >= MAX_STEPS;
    bool done = terminated || truncated;

    // Reward: penalize distance from target, velocity, and current usage
    env->rewards[0] = done ? 0.0f : 5.0f - fabsf(env->x - X_TARGET) - fabsf(env->v) / V_MAX - fabsf(a) / I_MAX;
    env->terminals[0] = terminated ? 1 : 0;

    if (env->truncations) env->truncations[0] = truncated ? 1 : 0;

    if (done) {
        add_log(env);
        c_reset(env);
    } else {
        // Update observations
        env->observations[0] = env->x;
        env->observations[1] = env->v;
    }
}

// Required function. Should handle creating the client on first call
void c_render(MagLev* env) {
    if (!IsWindowReady()) {
        InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "PufferLib MagLev");
        SetTargetFPS(30);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    // Draw floor/base
    int floor_y = WINDOW_HEIGHT - 60;
    DrawRectangle(0, floor_y, WINDOW_WIDTH, 60, (Color){80, 80, 80, 255});
    // Draw ball
    int ball_x = WINDOW_WIDTH / 2;
    // Map env->x (Y_MIN..Y_MAX) to screen y (floor_y..top)
    float y_norm = (env->x - Y_MIN) / (Y_MAX - Y_MIN);
    int ball_y = floor_y - (int)(y_norm * (floor_y - BALL_RADIUS));
    DrawCircle(ball_x, ball_y, BALL_RADIUS, (Color){0, 187, 187, 255});
    // Optionally: draw target line
    float target_norm = (X_TARGET - Y_MIN) / (Y_MAX - Y_MIN);
    int target_y = floor_y - (int)(target_norm * (floor_y - BALL_RADIUS));
    DrawLine(0, target_y, WINDOW_WIDTH, target_y, (Color){187, 0, 0, 255});
    DrawText("Target", 10, target_y - 20, 20, (Color){187, 0, 0, 255});
    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(MagLev* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
