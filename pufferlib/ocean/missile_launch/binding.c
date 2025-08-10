#include "missile_launch.h"

#define Env MissileLaunch
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    //env->size = unpack(kwargs, "size");
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
