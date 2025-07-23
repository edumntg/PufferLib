#include "maglev.h"

int main() {
    MagLev env = {.size = 2};
    env.observations = (float*)calloc(4, sizeof(float));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = -1.0f; // zero current when no normalized
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] += 0.1f;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] -= 0.1f;
            if (IsKeyDown(KEY_RIGHT)) env.m += 0.1f;
            if (IsKeyDown(KEY_LEFT)) env.m -= 0.1f;
        } else {
            env.actions[0] = rand() % 2.0f;
        }
        c_step(&env);
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}