//
// Created by Eduardo Montilva on 24/7/25.
//

#include "hanoi.h"

int main() {
    Hanoi env = {.size = 2};
    env.observations = (float*)calloc(env->num_disks * env->num_pegs, sizeof(float));
    env.actions = (int*)calloc(2, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env.disk_colors = (Color*)malloc(env->num_disks * sizeof(Color));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        // Randomly select two different pegs for from_peg and to_peg
        env.actions[0] = rand() % env->num_pegs;
        do {
            env.actions[1] = rand() % env->num_pegs;
        } while (env.actions[1] == env.actions[0]);
        c_step(&env);
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}