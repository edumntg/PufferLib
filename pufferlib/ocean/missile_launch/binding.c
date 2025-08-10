#include "missile_launch.h"

#define Env MissileLaunch
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->num_agents = unpack(kwargs, "num_agents");
    env->position = calloc(env->num_agents, sizeof(Vec2));
    env->velocity = calloc(env->num_agents, sizeof(Vec2));
    env->theta = calloc(env->num_agents, sizeof(float));
    env->omega = calloc(env->num_agents, sizeof(float));
    env->delta = calloc(env->num_agents, sizeof(float));
    env->prev_distance = calloc(env->num_agents, sizeof(float));
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
