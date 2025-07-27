//
// Created by Eduardo Montilva on 20/7/25.
//

#include <math.h>
#include <stdbool.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "raylib.h"

#define G 9.81f // gravity constant
#define PI 3.14159265358979323846f // pi constant
#define DT 0.02f // time step for simulation

// Obs. space bounds
#define Y_MIN 0.0f
#define Y_MAX 0.7f
#define V_MIN -3.0f
#define V_MAX 3.0f
#define M_MIN 0.1f
#define M_MAX 0.5f

// Action space bounds
#define I_MIN 0.0f
#define I_MAX 4.0f

#define MAX_STEPS 200 // max sim steps (MAX_STEPS * DT = seconds)
#define eps 0.0001f // epsilon for floating point comparison
#define K 0.08f // spring constant for magnetic force
#define BALL_RADIUS 32 // Ball radius in pixels
#define WINDOW_WIDTH 400
#define WINDOW_HEIGHT 600
#define DAMPING_COEFF 0.5f // damping coefficient for velocity

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
    float target; // target

    float m; // mass

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

float random_float(float low, float high) {
    return low + (high - low) * ((float)rand() / (float)RAND_MAX);
}

// Reset environment to initial state
void c_reset(MagLev* env) {
    env->x = random_float(-0.6f, 0.3f); // start at any position but at the bottom
    env->v = 0.0f; // start almost at rest


    env->target = random_float(-0.5f, 0.5f);

    env->m = random_float(-1.0f, 1.0f);

    env->tick = 0;
    env->size = 1; // Default size for window scaling (can be set by binding)

    env->actions[0] = random_float(-1.0f, 1.0f); // Random initial current
    env->observations[0] = env->x; // position
    env->observations[1] = env->v; // velocity
    env->observations[2] = fabsf(env->x - env->target); // distance to target
    env->observations[3] = env->m;

    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    if (env->truncations) env->truncations[0] = 0;
}

float clamp(float value, float min, float max) {
    return fminf(fmaxf(value, min), max);
}

float normalize(float value, float min, float max) {
    return -1.0f + (value - min) * 2.0f / (max - min);
}

float denormalize(float value, float min, float max) {
    return (value + 1.0f) * (max - min) * 0.5f + min;
}

// Step environment forward
void c_step(MagLev* env) {
    float a = env->actions[0];

    if (!isfinite(a)) {
        a = -1.0f;
    }

    // Clamp in case it gets out of bounds by manual input
    a = clamp(a, -1.0f, 1.0f);

    env->actions[0] = a;
    // Magnetic force: F = k * i^2 / (x + eps)^2
    float i = denormalize(a, I_MIN, I_MAX); //(a + 1.0f) * (I_MAX - I_MIN) * 0.5f + I_MIN;
    // compute real position and velocity (de-scaled)
    float x = denormalize(env->x, Y_MIN, Y_MAX); //(env->x + 1.0f) * (Y_MAX - Y_MIN) * 0.5f + Y_MIN;
    float v = denormalize(env->v, V_MIN, V_MAX); //(env->v + 1.0f) * (V_MAX - V_MIN) * 0.5f + V_MIN;
    float m = denormalize(env->m, M_MIN, M_MAX); //(env->m + 1.0f) * (M_MAX - M_MIN) * 0.5f + M_MIN;

    float dist = fmaxf(x, eps);  // Prevent division by zero
    float balance = K * i * i / (dist * dist) - m * G - DAMPING_COEFF * v;

    // Compute changes
    float v_dot = (balance / m);
    float x_dot = v;

    // Update values
    x += DT * x_dot;
    v += DT * v_dot;

    // Normalize and store
    env->x = normalize(x, Y_MIN, Y_MAX); //-1.0f + (x - Y_MIN) * 2.0f / (Y_MAX - Y_MIN);
    env->v = normalize(v, V_MIN, V_MAX); //-1.0f + (v - V_MIN) * 2.0f / (V_MAX - V_MIN);

    env->tick += 1;

    // Termination and truncation
    bool terminated = env->x < -1.0f || env->x > 1.0f || env->v < -1.0f || env->v > 1.0f;
    bool truncated = env->tick >= MAX_STEPS;
    bool done = terminated || truncated;

    // Reward: penalize distance from target, velocity, and current usage
    //env->rewards[0] = terminated ? -1.0f : Y_MAX - fabsf(env->x - Y_TARGET) - fabsf(env->v) / V_MAX - fabsf(a) / I_MAX;
    float dist_penalty = 2.0f*denormalize(fabsf(env->x - env->target), Y_MIN, Y_MAX);
    float vel_penalty = 0.1f * denormalize(fabsf(env->v), V_MIN, V_MAX);
    float action_penalty = 0.01f * denormalize(fabsf(env->actions[0]), I_MIN, I_MAX);

    // Only padd a high penalty if the sim is terminated (out of window, too much velocity, etc)
    // If sim is ended because of max_steps, it means that the ball remained in the window so just add a penalty
    // based on the distance to the target, velocity and action
    env->rewards[0] = terminated ? -1.0f : 1.0f -dist_penalty - vel_penalty;

    env->terminals[0] = terminated ? 1 : 0;

    if (env->truncations) env->truncations[0] = truncated ? 1 : 0;

    if (done) {
        add_log(env);
        c_reset(env);
    } else {
        // Update observations
        env->observations[0] = env->x;
        env->observations[1] = env->v;
        env->observations[2] = fabsf(env->x - env->target);
        env->observations[3] = env->m;
    }
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(MagLev* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}


// Required function. Should handle creating the client on first call
void c_render(MagLev* env) {
    if (!IsWindowReady()) {
        InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "PufferLib MagLev");
        SetTargetFPS(30);
    }

    if (WindowShouldClose()) {
        c_close(env);
        exit(0);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    if (IsKeyDown(KEY_LEFT_SHIFT)) {
        if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env->actions[0] += 0.1f;
        if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env->actions[0] -= 0.1f;
        if (IsKeyDown(KEY_RIGHT)) env->m += 0.1f;
        if (IsKeyDown(KEY_LEFT)) env->m -= 0.1f;
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    // Draw floor/base
    int floor_y = WINDOW_HEIGHT - 60;
    DrawRectangle(0, floor_y, WINDOW_WIDTH, 60, (Color){80, 80, 80, 255});

    // Env values
    float i = denormalize(env->actions[0], I_MIN, I_MAX); //(env->actions[0] + 1.0f) * (I_MAX - I_MIN) * 0.5f + I_MIN;
    float x = denormalize(env->x, Y_MIN, Y_MAX); //(env->observations[0] + 1.0f) * (Y_MAX - Y_MIN) * 0.5f + Y_MIN;
    float v = denormalize(env->v, V_MIN, V_MAX); //(env->observations[1] + 1.0f) * (V_MAX - V_MIN) * 0.5f + V_MIN;

    float target = denormalize(env->target, Y_MIN, Y_MAX); //(env->target + 1.0f) * (Y_MAX - Y_MIN) * 0.5f + Y_MIN;
    float m = denormalize(env->m, M_MIN, M_MAX); //

    float distance = fabsf(x - target);

    // Draw ball
    int ball_x = WINDOW_WIDTH / 2;
    float y_norm = (x - Y_MIN) / (Y_MAX - Y_MIN);
    int ball_y = floor_y - (int)(y_norm * (floor_y - BALL_RADIUS));
    DrawCircle(ball_x, ball_y, BALL_RADIUS, (Color){0, 187, 187, 255});

    // Draw target line
    float target_norm = (target - Y_MIN) / (Y_MAX - Y_MIN);
    int target_y = floor_y - (int)(target_norm * (floor_y - BALL_RADIUS));
    DrawLine(0, target_y, WINDOW_WIDTH, target_y, (Color){187, 0, 0, 255});
    DrawText("Target", 10, target_y - 20, 20, (Color){187, 0, 0, 255});

    // Render current action value as "I = <value>"
    char current_text[32];
    snprintf(current_text, sizeof(current_text), "I = %.4f", i);
    DrawText(current_text, 10, 10, 20, (Color){255, 255, 255, 255});

    // Render current position and velocity
    char position_text[64];
    snprintf(position_text, sizeof(position_text), "Position: %.2f m", x);
    DrawText(position_text, 10, 40, 20, (Color){255, 255, 255, 255});

    char velocity_text[64];
    snprintf(velocity_text, sizeof(velocity_text), "Velocity: %.2f m/s", v);
    DrawText(velocity_text, 10, 70, 20, (Color){255, 255, 255, 255});

    char target_text[64];
    snprintf(target_text, sizeof(target_text), "Target: %.2f m", target);
    DrawText(target_text, 10, 100, 20, (Color){255, 255, 255, 255});

    char distance_text[64];
    snprintf(distance_text, sizeof(distance_text), "Distance: %.2f m", distance);
    DrawText(distance_text, 10, 130, 20, (Color){255, 255, 255, 255});

    char mass_text[64];
    snprintf(mass_text, sizeof(mass_text), "Mass: %.2f kg", m);
    DrawText(mass_text, 10, 160, 20, (Color){255, 255, 255, 255});

    char reward_text[64];
    snprintf(reward_text, sizeof(reward_text), "Reward: %.4f", env->rewards[0]);
    DrawText(reward_text, 10, 190, 20, (Color){255, 255, 255, 255});

    EndDrawing();
}