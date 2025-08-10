#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include "math.h"

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

// Physics params
const float ROCKET_LENGTH = 3.0f;
const float ROCKET_THRUST = 800.0f;  // magnitude
const float GRAVITY = 9.81f;
const float ROCKET_MASS = 20.0f;
const float I = (1/12.0f) * ROCKET_MASS * (ROCKET_LENGTH * ROCKET_LENGTH); // MoI rod

// Controls/limits
const float MAX_GIMBAL_ANGLE = 10.0f; // deg
const float DT = 0.05f;
const int MAX_STEPS = 600;

// Stabilization/shaping
const float ANGULAR_DAMPING = 2.0f;   // per-second damping on omega
const float LINEAR_DAMPING  = 0.2f;   // per-second damping on v
const float TORQUE_GAIN     = 0.2f;   // scale torque to reduce spin
const float HIT_RADIUS      = 40.0f;  // pixels

const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;

// Only use floats!
typedef struct {
    float score;
    float n; // Required as the last field
} Log;

typedef struct {
    float x;
    float y;
} Vec2;

static inline Vec2 add2(Vec2 a, Vec2 b) { return (Vec2){a.x + b.x, a.y + b.y}; }
static inline Vec2 sub2(Vec2 a, Vec2 b) { return (Vec2){a.x - b.x, a.y - b.y}; }
static inline Vec2 scalmul2(Vec2 a, float b) { return (Vec2){a.x * b, a.y * b}; }
static inline float dot2(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
static inline float norm2(Vec2 a) { return sqrtf(dot2(a, a)); }

static inline float clampf(float v, float min, float max) {
    if (v < min) return min;
    if (v > max) return max;
    return v;
}

static inline float wrap_pi(float a) {
    while (a >  M_PI) a -= 2.0f*M_PI;
    while (a < -M_PI) a += 2.0f*M_PI;
    return a;
}

typedef struct {
    Log log;                     // Required field
    float* observations;         // Required field. Ensure type matches in .py and .c
    float* actions;              // Required field. Ensure type matches in .py and .c
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field

    Vec2 position;
    Vec2 velocity;
    Vec2 target; // pos

    float theta; // pitch
    float omega; // angular velocity
    float delta;

    float prev_distance;
    int tick;
} MissileLaunch;

static inline void write_observations(MissileLaunch* env) {
    // Observations:
    // 0: dx, 1: dy, 2: vx, 3: vy, 4: angle_error, 5: omega
    float dx = env->position.x - env->target.x;
    float dy = env->position.y - env->target.y;
    float angle_to_target = atan2f(env->target.y - env->position.y,
                                   env->target.x - env->position.x);
    float angle_error = wrap_pi(angle_to_target - env->theta);

    env->observations[0] = dx;
    env->observations[1] = dy;
    env->observations[2] = env->velocity.x;
    env->observations[3] = env->velocity.y;
    env->observations[4] = angle_error;
    env->observations[5] = env->omega;
}

void c_reset(MissileLaunch* env) {
    env->position.x = (float)(rand() % (SCREEN_WIDTH / 2));
    env->position.y = 30.0f;
    env->velocity.x = 0.0f;
    env->velocity.y = 0.0f;
    env->theta = 0.0f;
    env->omega = 0.0f;

    // Random target pos in right half, avoid floor
    env->target.x = (float)(rand() % (SCREEN_WIDTH / 2));
    env->target.y = (float)(rand() % (SCREEN_HEIGHT - 30));

    env->tick = 0;

    // Initialize distance and observations
    env->prev_distance = norm2(sub2(env->position, env->target));
    write_observations(env);
}

void c_step(MissileLaunch* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    env->tick += 1;

    // Action → gimbal angle in radians
    float thrust = env->actions[0] * ROCKET_THRUST; // scale thrust
    float delta = env->actions[1];
    delta = clampf(delta, -1.0f, 1.0f);
    delta = delta * MAX_GIMBAL_ANGLE * (M_PI / 180.0f);

    env->delta = delta;

    // Torque and angular dynamics with damping
    float torque = TORQUE_GAIN * (ROCKET_LENGTH / 2.0f) * thrust * sinf(delta);
    float alpha = torque / I;
    env->omega += (alpha - ANGULAR_DAMPING * env->omega) * DT;
    env->theta = wrap_pi(env->theta + env->omega * DT);

    // Thrust components (body angle + gimbal) and gravity
    float thrust_x = thrust * sinf(env->theta + delta);
    float thrust_y = thrust * cosf(env->theta + delta) - (GRAVITY * ROCKET_MASS);
    float ax = thrust_x / ROCKET_MASS;
    float ay = thrust_y / ROCKET_MASS;

    // Linear dynamics + damping
    env->velocity.x += ax * DT;
    env->velocity.y += ay * DT;
    env->velocity.x -= LINEAR_DAMPING * env->velocity.x * DT;
    env->velocity.y -= LINEAR_DAMPING * env->velocity.y * DT;

    env->position.x += env->velocity.x * DT;
    env->position.y += env->velocity.y * DT;

    // Check target hit
    float dist_to_target = norm2(sub2(env->position, env->target));
    if (dist_to_target < HIT_RADIUS) {
        env->rewards[0] = 1.0f;
        env->terminals[0] = 1;
        env->log.score += 1.0f;
        env->log.n += 1.0f;
        c_reset(env);
        return;
    }

    // Bounds / timeout
    bool out_of_bounds = env->position.x < -10.0f || env->position.x > SCREEN_WIDTH + 10.0f
      || env->position.y < -10.0f || env->position.y > SCREEN_HEIGHT + 10.0f;

    if (env->tick >= MAX_STEPS || out_of_bounds) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0f;
        env->log.n += 1.0f;
        env->log.score += env->rewards[0];
        c_reset(env);
        return;
    }

    // Compute unit vector to target
    Vec2 to_target = sub2(env->target, env->position);
    float dist = fmaxf(norm2(to_target), 1e-3f);
    Vec2 to_target_unit = scalmul2(to_target, 1.0f / dist);

    // Radial speed (positive if moving toward target)
    float radial_speed = dot2(env->velocity, to_target_unit);

    // Angle alignment in [-1, 1]
    float angle_to_target = atan2f(to_target.y, to_target.x);
    float angle_error = wrap_pi(angle_to_target - env->theta);
    float align = cosf(angle_error);

    // Scaled, lower-variance reward
    env->rewards[0] = 0.01f * radial_speed      // move toward target
                    + 0.002f * align            // point toward target
                    - 0.0002f * dist            // gentle pull to be closer
                    - 0.001f * fabsf(env->omega);

    env->prev_distance = dist_to_target;

    // Observations
    write_observations(env);
}

void c_close(MissileLaunch* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

void c_render(MissileLaunch* env) {
    if (!IsWindowReady()) {
        InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Missile Launch");
        SetTargetFPS(30);
    }

    if (WindowShouldClose()) {
        c_close(env);
        exit(0);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        c_close(env);
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    // Draw surface/floor
    DrawRectangle(0, SCREEN_HEIGHT - 30, SCREEN_WIDTH, 30, PUFF_WHITE);

    // Draw Rocket
    int rocket_render_width = 20;
    int rocket_render_length = 50;
    Rectangle rect = {env->position.x, SCREEN_HEIGHT - env->position.y, rocket_render_width, rocket_render_length};
    Vector2 origin = {rocket_render_width / 2, rocket_render_length}; // center

    DrawRectanglePro(rect, origin, env->theta * 180.0f / M_PI, PUFF_CYAN);

    // Draw collision radius
    DrawCircle(env->position.x, SCREEN_HEIGHT - env->position.y, 5, PUFF_RED);
    DrawCircleLines(env->position.x, SCREEN_HEIGHT - env->position.y, HIT_RADIUS, (Color){255, 0, 0, 100});

    // Fire cone
    float delta_action = clampf(env->actions[0], -1.0f, 1.0f);
    float delta_rad = delta_action * MAX_GIMBAL_ANGLE * M_PI / 180.0f;

    Vector2 rocket_center = { env->position.x, SCREEN_HEIGHT - env->position.y };
    float rocket_angle_rad = env->theta;
    float combined_angle_rad = env->theta + delta_rad;

    float rocket_base_x = rocket_center.x - (rocket_render_length / 2.0f) * sinf(rocket_angle_rad);
    float rocket_base_y = rocket_center.y + (rocket_render_length / 4.0f) * cosf(rocket_angle_rad);
    Vector2 rocket_base = { rocket_base_x, rocket_base_y };

    float fire_length = 15.0f;
    float fire_width = 10.0f;

    Vector2 fire_tip = {
        rocket_base.x + fire_length * sinf(combined_angle_rad),
        rocket_base.y + fire_length * cosf(combined_angle_rad)
    };

    Vector2 base_left = {
        rocket_base.x + (fire_width / 2.0f) * cosf(combined_angle_rad),
        rocket_base.y - (fire_width / 2.0f) * sinf(combined_angle_rad)
    };

    Vector2 base_right = {
        rocket_base.x - (fire_width / 2.0f) * cosf(combined_angle_rad),
        rocket_base.y + (fire_width / 2.0f) * sinf(combined_angle_rad)
    };

    DrawTriangle(fire_tip, base_left, base_right, ORANGE);

    // Draw Target (fix y inversion)
    DrawCircle(env->target.x, SCREEN_HEIGHT - env->target.y, 5, PUFF_RED);

    char theta_text[64];
    snprintf(theta_text, sizeof(theta_text), "theta = %.2f deg", env->theta * 180.0f / M_PI);
    DrawText(theta_text, 10, 10, 20, (Color){255, 255, 255, 255});

    char velocity_text[64];
    snprintf(velocity_text, sizeof(velocity_text), "Vel: (%.2f, %.2f)", env->velocity.x, env->velocity.y);
    DrawText(velocity_text, 10, 40, 20, (Color){255, 255, 255, 255});

    char omega_text[64];
    snprintf(omega_text, sizeof(omega_text), "Omega = %.3f", env->omega);
    DrawText(omega_text, 10, 70, 20, (Color){255, 255, 255, 255});

    char steps_text[32];
    snprintf(steps_text, sizeof(steps_text), "Steps = %d", env->tick);
    DrawText(steps_text, 10, 100, 20, (Color){255, 255, 255, 255});

    char delta_text[32];
    snprintf(delta_text, sizeof(delta_text), "Delta = %.4f", env->delta * 180.0f / M_PI);
    DrawText(delta_text, 10, 130, 20, (Color){255, 255, 255, 255});

    char thrust_text[32];
    snprintf(thrust_text, sizeof(thrust_text), "Thrust = %.4f", env->actions[0] * ROCKET_THRUST);
    DrawText(thrust_text, 10, 160, 20, (Color){255, 255, 255, 255});

    char reward_text[32];
    snprintf(reward_text, sizeof(reward_text), "Reward = %.4f", env->rewards[0]);
    DrawText(reward_text, 10, 190, 20, (Color){255, 255, 255, 255});
    EndDrawing();
}