#include "hanoi.h"
#define Env Hanoi
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->size = 1; // default
    if (kwargs && PyDict_Check(kwargs)) {
        PyObject* disks_obj = PyDict_GetItemString(kwargs, "num_disks");
        if (disks_obj && PyLong_Check(disks_obj)) {
            env->num_disks = (int)PyLong_AsLong(disks_obj);
        }
        PyObject* pegs_obj = PyDict_GetItemString(kwargs, "num_pegs");
        if (pegs_obj && PyLong_Check(pegs_obj)) {
            env->num_pegs = (int)PyLong_AsLong(pegs_obj);
        }
    }
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}