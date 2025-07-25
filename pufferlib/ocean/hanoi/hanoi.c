//
// Created by Eduardo Montilva on 24/7/25.
//

#include "hanoi.h"

int main() {
    Hanoi env = {.size = 2};
    env.observations = (float*)calloc(NUM_DISKS * NUM_PEGS, sizeof(float));
    env.actions = (int*)calloc(2, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env.disk_colors = (Color*)malloc(NUM_DISKS * sizeof(Color));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        // Randomly select two different pegs for from_peg and to_peg
        env.actions[0] = rand() % NUM_PEGS;
        do {
            env.actions[1] = rand() % NUM_PEGS;
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