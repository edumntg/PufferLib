//
// Created by Eduardo Montilva on 20/7/25.
//

#include "maglev.h"

#define Env MagLev
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->size = 1; // default
    if (kwargs && PyDict_Check(kwargs)) {
        PyObject* size_obj = PyDict_GetItemString(kwargs, "size");
        if (size_obj && PyLong_Check(size_obj)) {
            env->size = (int)PyLong_AsLong(size_obj);
        }
    }
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
