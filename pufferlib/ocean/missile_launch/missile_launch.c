#include "template.h"

int main() {
    int num_agents = 16;
    MissileLaunch env;
    env.observations = (float*)calloc(6*num_agents, sizeof(float));
    env.actions = (float*)calloc(2*num_agents, sizeof(float));
    env.rewards = (float*)calloc(num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(num_agents, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        c_step(&env);
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

