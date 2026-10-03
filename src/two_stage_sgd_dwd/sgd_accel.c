#define Py_LIMITED_API 0x030B0000
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Keep NumPy's dot implementation and strict, separate scalar operations. */
typedef struct {
    PyObject *dot_function, *copy_function, *empty_function;
} Functions;

static int vector_buffer(PyObject *object, Py_buffer *view, int integer, int writable) {
    int flags = PyBUF_FORMAT | PyBUF_ND | PyBUF_C_CONTIGUOUS | (writable ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(object, view, flags) < 0) return 0;
    if (view->ndim != 1 || view->itemsize != 8 || view->format == NULL ||
            ((uintptr_t)view->buf % 8 != 0) ||
            (integer ? (strcmp(view->format, "q") != 0 && strcmp(view->format, "l") != 0) : strcmp(view->format, "d") != 0)) {
        PyBuffer_Release(view);
        view->obj = NULL;
        PyErr_SetString(PyExc_ValueError, "Expected an aligned contiguous float64 or int64 vector");
        return 0;
    }
    return 1;
}

static int numpy_dot(PyObject *function, PyObject *left, PyObject *right, double *value) {
    PyObject *answer = PyObject_CallFunctionObjArgs(function, left, right, NULL);
    if (answer == NULL) return 0;
    *value = PyFloat_AsDouble(answer);
    Py_DECREF(answer);
    return !PyErr_Occurred();
}

static int take_numeric_error(PyObject **error) {
    if (!(PyErr_ExceptionMatches(PyExc_FloatingPointError) || PyErr_ExceptionMatches(PyExc_OverflowError))) return 0;
    PyObject *type = NULL, *value = NULL, *trace = NULL;
    PyErr_Fetch(&type, &value, &trace);
    PyErr_NormalizeException(&type, &value, &trace);
    *error = value ? PyObject_Str(value) : PyUnicode_FromString("Numerical operation failed");
    Py_XDECREF(type); Py_XDECREF(value); Py_XDECREF(trace);
    return *error != NULL;
}

static PyObject *run_steps(PyObject *self, PyObject *args) {
    Functions *functions = (Functions *)PyModule_GetState(self);
    int kind;
    double bias, regularization;
    PyObject *weights, *get_row, *labels, *indices, *rates;
    PyObject *current = NULL, *proposal = NULL, *row = NULL, *error = NULL, *answer = NULL;
    Py_buffer input = {0}, a = {0}, z = {0}, y = {0}, ix = {0}, eta = {0}, x = {0};
    Py_ssize_t completed = 0;
    if (!PyArg_ParseTuple(args, "iOdOOOOd", &kind, &weights, &bias, &get_row,
                          &labels, &indices, &rates, &regularization)) return NULL;
    if ((kind != 0 && kind != 1) || !isfinite(bias) || !isfinite(regularization) ||
            (kind == 1 && regularization <= 0) || !PyCallable_Check(get_row)) {
        PyErr_SetString(PyExc_ValueError, "Invalid scalar solver controls");
        return NULL;
    }
    if (!vector_buffer(weights, &input, 0, 0)) goto cleanup;
    current = PyObject_CallFunctionObjArgs(functions->copy_function, weights, NULL);
    if (current == NULL) goto cleanup;
    proposal = PyObject_CallFunctionObjArgs(functions->empty_function, current, NULL);
    if (proposal == NULL || !vector_buffer(current, &a, 0, 1) || !vector_buffer(proposal, &z, 0, 1) ||
            !vector_buffer(labels, &y, 0, 0) || !vector_buffer(indices, &ix, 1, 0) ||
            !vector_buffer(rates, &eta, 0, 0)) goto cleanup;
    Py_ssize_t width = a.len / 8, count = ix.len / 8, n = y.len / 8;
    if (eta.len != ix.len || z.len != a.len || width == 0 || (kind == 1 && width != n)) {
        PyErr_SetString(PyExc_ValueError, "Mismatched scalar solver dimensions");
        goto cleanup;
    }
    const double *ys = (const double *)y.buf, *steps = (const double *)eta.buf;
    const int64_t *samples = (const int64_t *)ix.buf;
    for (Py_ssize_t j = 0; j < width; j++) if (!isfinite(((double *)a.buf)[j])) {
        PyErr_SetString(PyExc_ValueError, "Nonfinite initial coefficients"); goto cleanup;
    }
    for (Py_ssize_t t = 0; t < count; t++) {
        int64_t i = samples[t];
        if (i < 0 || i >= n || !isfinite(steps[t]) || steps[t] < 0 || (ys[i] != -1 && ys[i] != 1)) {
            PyErr_SetString(PyExc_ValueError, "Invalid sampled observation or step size"); goto cleanup;
        }
        PyObject *index = PyLong_FromLongLong(i);
        if (index == NULL) goto cleanup;
        row = PyObject_CallFunctionObjArgs(get_row, index, NULL);
        Py_DECREF(index);
        if (row == NULL) {
            if (take_numeric_error(&error)) break;
            goto cleanup;
        }
        if (!vector_buffer(row, &x, 0, 0)) goto cleanup;
        if (x.len != a.len) { PyErr_SetString(PyExc_ValueError, "Sampled vector has wrong length"); goto cleanup; }
        double dot, margin, q, multiplier, new_bias, shrink = 0;
        if (!numpy_dot(functions->dot_function, current, row, &dot)) {
            if (take_numeric_error(&error)) goto numerical_end;
            goto cleanup;
        }
        margin = ys[i] * (dot + bias);
        if (!isfinite(margin)) { error = PyUnicode_FromString("Nonfinite scalar margin"); goto numerical_end; }
        q = margin > 0.5 ? (-0.25 / margin) / margin : -1.0;
        q *= ys[i];
        multiplier = steps[t] * q;
        new_bias = bias - multiplier;
        if (!isfinite(new_bias) || !isfinite(multiplier)) { error = PyUnicode_FromString("Nonfinite scalar update"); goto numerical_end; }
        double *old = (double *)a.buf, *next = (double *)z.buf, *features = (double *)x.buf;
        if (kind == 1) {
            shrink = 1.0 - steps[t] * (2.0 * regularization);
            if (!isfinite(shrink)) { error = PyUnicode_FromString("Nonfinite shrinkage"); goto numerical_end; }
            for (Py_ssize_t j = 0; j < width; j++) next[j] = shrink * old[j];
            next[i] -= multiplier;
        } else {
            double scale = 0;
            for (Py_ssize_t j = 0; j < width; j++) {
                double increment = multiplier * features[j];
                next[j] = old[j] - increment;
                if (!isfinite(next[j])) { error = PyUnicode_FromString("Nonfinite linear proposal"); goto numerical_end; }
                double magnitude = fabs(next[j]);
                if (magnitude > scale) scale = magnitude;
            }
            if (scale > 1.0) for (Py_ssize_t j = 0; j < width; j++) next[j] /= scale;
            double squared_norm;
            if (!numpy_dot(functions->dot_function, proposal, proposal, &squared_norm)) {
                if (take_numeric_error(&error)) goto numerical_end;
                goto cleanup;
            }
            double norm = sqrt(squared_norm);
            if (scale > 1.0 || norm > 1.0) for (Py_ssize_t j = 0; j < width; j++) next[j] /= norm;
        }
        for (Py_ssize_t j = 0; j < width; j++) if (!isfinite(next[j])) {
            error = PyUnicode_FromString("Nonfinite coefficients"); goto numerical_end;
        }
        bias = new_bias;
        { PyObject *swap = current; current = proposal; proposal = swap;
          Py_buffer swap_view = a; a = z; z = swap_view; }
        completed++;
numerical_end:
        PyBuffer_Release(&x); x.obj = NULL;
        Py_CLEAR(row);
        if (PyErr_Occurred()) goto cleanup;
        if (error != NULL) break;
    }
    answer = Py_BuildValue("OdnO", current, bias, completed, error ? error : Py_None);
cleanup:
    if (input.obj) PyBuffer_Release(&input);
    if (x.obj) PyBuffer_Release(&x);
    if (a.obj) PyBuffer_Release(&a);
    if (z.obj) PyBuffer_Release(&z);
    if (y.obj) PyBuffer_Release(&y);
    if (ix.obj) PyBuffer_Release(&ix);
    if (eta.obj) PyBuffer_Release(&eta);
    Py_XDECREF(row); Py_XDECREF(current); Py_XDECREF(proposal); Py_XDECREF(error);
    return answer;
}

static PyMethodDef methods[] = {
    {"run_steps", run_steps, METH_VARARGS, "Execute ordered scalar updates using NumPy dot and strict arithmetic."},
    {NULL, NULL, 0, NULL}
};
static int module_traverse(PyObject *module, visitproc visit, void *arg) {
    Functions *state = (Functions *)PyModule_GetState(module);
    Py_VISIT(state->dot_function); Py_VISIT(state->copy_function); Py_VISIT(state->empty_function);
    return 0;
}
static int module_clear(PyObject *module) {
    Functions *state = (Functions *)PyModule_GetState(module);
    Py_CLEAR(state->dot_function); Py_CLEAR(state->copy_function); Py_CLEAR(state->empty_function);
    return 0;
}
static void module_free(void *module) { (void)module_clear((PyObject *)module); }
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_sgd_accel", NULL, sizeof(Functions), methods,
    NULL, module_traverse, module_clear, module_free
};
PyMODINIT_FUNC PyInit__sgd_accel(void) {
    PyObject *created = PyModule_Create(&module);
    if (created == NULL) return NULL;
    Functions *state = (Functions *)PyModule_GetState(created);
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) { Py_DECREF(created); return NULL; }
    state->dot_function = PyObject_GetAttrString(numpy, "dot");
    state->copy_function = PyObject_GetAttrString(numpy, "copy");
    state->empty_function = PyObject_GetAttrString(numpy, "empty_like");
    Py_DECREF(numpy);
    if (!state->dot_function || !state->copy_function || !state->empty_function) {
        Py_DECREF(created); return NULL;
    }
    return created;
}
