#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include "math.h"

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

// ISU values
const float ROCKET_LENGTH = 3.0f;
const float ROCKET_THRUST = 800.0f;
const float GRAVITY = 9.81f;
const float ROCKET_MASS = 20.0f;

const float MAX_GIMBAL_ANGLE = 10.0f;

const float DT = 0.05f;

const float I = (1/12.0f) * ROCKET_MASS * (ROCKET_LENGTH * ROCKET_LENGTH); // Moment of inertia of a rod about its center

const int MAX_STEPS = 1000;

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
    if (v < min)
        return min;
    if (v > max)
        return max;
    return v;
}

typedef struct {
    Log log;                     // Required field
    float* observations; // Required field. Ensure type matches in .py and .c
    float* actions;                // Required field. Ensure type matches in .py and .c
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field

    Vec2 position;
    Vec2 velocity;
    Vec2 target; // pos

    float theta; // pitch
    float omega; // angular velocity

    int tick;

} MissileLaunch;

void c_reset(MissileLaunch* env) {
    // Reset the position and target
    env->position.x = 30.0f;
    env->position.y = 30.0f;
    env->velocity.x = 0.0f;
    env->velocity.y = 0.0f;
    env->theta = 0.0f;
    env->omega = 0.0f;

    // Random target pos between the center of the screen to the right
	env->target.x = (float)(rand() % (SCREEN_WIDTH / 2) + SCREEN_WIDTH / 2);
	env->target.y = (float)(rand() % (SCREEN_HEIGHT - 30)); // Avoid the floor

    env->tick = 0;
}

void c_step(MissileLaunch* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;

    env->tick += 1;

    // Get gimbal angle
    float delta = env->actions[0];

    // Clamp
    delta = clampf(delta, -1.0f, 1.0f);

    // De-normalize (Convert to degrees)
    delta = delta * MAX_GIMBAL_ANGLE; // This is in degrees

    // Convert delta to radians for calculations
    delta = delta * M_PI / 180.0f;

    // Torque
    float torque = (ROCKET_LENGTH / 2.0f) * ROCKET_THRUST * sinf(delta);

    // Angular acceleration
    float alpha = torque / I;

    // Angular velocity
    env->omega += alpha * DT;
    env->theta += env->omega * DT;

    if(env->theta > 2.0f*M_PI) {
        env->theta -= 2.0*M_PI;
    }
    if(env->theta < -2.0f*M_PI) {
        env->theta += 2.0*M_PI;
    }

    // X-Y params
    float thrust_x = ROCKET_THRUST * sinf(env->theta + delta);
    float thrust_y = ROCKET_THRUST * cosf(env->theta + delta) - (GRAVITY * ROCKET_MASS);
    float acceleration_x = thrust_x / ROCKET_MASS;
    float acceleration_y = thrust_y / ROCKET_MASS;

    // Update velocity
    env->velocity.x += acceleration_x * DT;
    env->velocity.y += acceleration_y * DT;

    // Update position
    env->position.x += env->velocity.x * DT;
    env->position.y += env->velocity.y * DT;
    
    // Now check if we hit the target
    float dist_to_target = norm2(sub2(env->position, env->target));
    if (dist_to_target < 1.0f) {
        // Hit the target
        env->rewards[0] = 1.0f;
        env->terminals[0] = 1;
        env->log.score += 1.0f;
        env->log.n += 1.0f;

        // Reset
        c_reset(env);
        return;
    }

    // Check if it is out of bounds
    bool out_of_bounds = env->position.x < -10.0f || env->position.x > SCREEN_WIDTH + 10.0f || env->position.y < -10.0f || env->position.y > SCREEN_HEIGHT + 10.0f;

    if(env->tick >= MAX_STEPS || out_of_bounds) {
        env->terminals[0] = 1; // Episode ends
        env->rewards[0] = -1.0f; // Negative reward for not hitting target in time
        env->log.n += 1;
        env->log.score += env->rewards[0]; // Add the negative reward to the score
        c_reset(env);
        return;
    }

    // No hit, so no reward
    env->rewards[0] = -0.5f * dist_to_target; // Negative reward based on distance to target
    env->terminals[0] = 0; // Not terminal

    // Update observations
    env->observations[0] = env->position.x - env->target.x; // Relative x position to target
    env->observations[1] = env->position.y - env->target.y;
    env->observations[2] = env->velocity.x; // Velocity x
    env->observations[3] = env->velocity.y;
    env->observations[4] = env->theta; // Pitch angle
    env->observations[5] = env->omega; // Angular velocity
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
    Vector2 origin = {rocket_render_width / 2, rocket_render_length / 2}; // center

    DrawRectanglePro(rect, origin, env->theta * 180.0f / M_PI, PUFF_CYAN);;

    // Draw a yellow smaller "fire"/cone at bottom of the rocket rotates based on the delta angle
    // Get the gimbal angle from the actions array
    float delta_action = env->actions[0];
    delta_action = clampf(delta_action, -1.0f, 1.0f);
    float delta_rad = delta_action * MAX_GIMBAL_ANGLE * M_PI / 180.0f; // Convert gimbal angle to radians

    // Rocket's position on the screen
    Vector2 rocket_center = { env->position.x, SCREEN_HEIGHT - env->position.y };

    // The rocket body's orientation angle is env->theta (already in radians)
    float rocket_angle_rad = env->theta;

    // The thrust vector's orientation angle, which determines the fire's direction
    float combined_angle_rad = env->theta + delta_rad;

    // Calculate the position of the rocket's base in screen coordinates
    // The vector from center to base is (-L/2) rotated by rocket_angle_rad.
    // Y is inverted for screen coordinates.
    float rocket_base_x = rocket_center.x - (rocket_render_length / 2.0f) * sinf(rocket_angle_rad);
    float rocket_base_y = rocket_center.y + (rocket_render_length / 2.0f) * cosf(rocket_angle_rad);
    Vector2 rocket_base = { rocket_base_x, rocket_base_y };

    // Define the fire cone dimensions and position relative to the rocket's base
    float fire_length = 15.0f;
    float fire_width = 10.0f;

    // Calculate the points of the triangle representing the fire.
    // The fire cone points away from the base in the direction of thrust.
    // Screen thrust vector: (sin(angle), cos(angle))
    Vector2 fire_tip = {
        rocket_base.x + fire_length * sinf(combined_angle_rad),
        rocket_base.y + fire_length * cosf(combined_angle_rad)
    };

    // The base of the fire triangle is perpendicular to the thrust vector.
    // Screen perpendicular vector: (cos(angle), -sin(angle))
    Vector2 base_left = {
        rocket_base.x + (fire_width / 2.0f) * cosf(combined_angle_rad),
        rocket_base.y - (fire_width / 2.0f) * sinf(combined_angle_rad)
    };

    Vector2 base_right = {
        rocket_base.x - (fire_width / 2.0f) * cosf(combined_angle_rad),
        rocket_base.y + (fire_width / 2.0f) * sinf(combined_angle_rad)
    };

    // Draw the triangle
    DrawTriangle(fire_tip, base_left, base_right, ORANGE);

    // Draw Target
    DrawCircle(env->target.x, env->target.y, 5, PUFF_RED);

    char theta_text[32];
    snprintf(theta_text, sizeof(theta_text), "theta = %.4f", env->theta * 180.0f / M_PI); // Convert to degrees for display
    DrawText(theta_text, 10, 10, 20, (Color){255, 255, 255, 255});

    // Now show velocity omega and pitch
    char velocity_text[64];
    snprintf(velocity_text, sizeof(velocity_text), "Velocity: (%.2f, %.2f)", env->velocity.x, env->velocity.y);
    DrawText(velocity_text, 10, 40, 20, (Color){255, 255, 255, 255});

    char omega_text[32];
    snprintf(omega_text, sizeof(omega_text), "Omega = %.4f", env->omega);
    DrawText(omega_text, 10, 70, 20, (Color){255, 255, 255, 255});

    char steps_text[32];
    snprintf(steps_text, sizeof(steps_text), "Steps = %d", env->tick);
    DrawText(steps_text, 10, 100, 20, (Color){255, 255, 255, 255});

    EndDrawing();
}