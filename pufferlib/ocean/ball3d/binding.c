#include "ball3d.h"

#define Env Ball3D
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->num_agents = unpack(kwargs, "num_agents");
    env->pos = calloc(env->num_agents, sizeof(Vec3));
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
