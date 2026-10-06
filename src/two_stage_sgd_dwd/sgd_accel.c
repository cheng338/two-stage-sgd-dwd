#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <fenv.h>

/* Retain NumPy dispatch unless a known SciPy C wrapper is usable. */
typedef double (*DDotFunction)(int *, double *, int *, double *, int *);
typedef struct {
    PyObject *dot_function, *copy_function, *empty_function, *ndarray_type;
    PyObject *blas_module, *ddot_capsule;
    DDotFunction ddot;
} Functions;

static int resolve_ddot(Functions *state) {
    const char *signature = "__pyx_t_5scipy_6linalg_11cython_blas_d (int *, "
        "__pyx_t_5scipy_6linalg_11cython_blas_d *, int *, "
        "__pyx_t_5scipy_6linalg_11cython_blas_d *, int *)";
    PyObject *multiarray = NULL, *canonical = NULL, *umath = NULL, *builtin = NULL;
    PyObject *dispatcher = NULL, *wrapped = NULL, *blas = NULL, *api = NULL, *capsule = NULL;
    int success = 1;
    multiarray = PyImport_ImportModule("numpy._core.multiarray");
    if (multiarray == NULL) goto optional_error;
    canonical = PyObject_GetAttrString(multiarray, "dot");
    if (canonical == NULL) goto optional_error;
    if (canonical != state->dot_function) goto done;
    umath = PyImport_ImportModule("numpy._core._multiarray_umath");
    if (umath == NULL) goto optional_error;
    dispatcher = PyObject_GetAttrString(umath, "_ArrayFunctionDispatcher");
    if (dispatcher == NULL) goto optional_error;
    if (!PyType_Check(dispatcher) || Py_TYPE(canonical) != (PyTypeObject *)dispatcher) goto done;
    builtin = PyObject_GetAttrString(umath, "dot");
    if (builtin == NULL) goto optional_error;
    wrapped = PyObject_GetAttrString(canonical, "__wrapped__");
    if (wrapped == NULL) goto optional_error;
    if (wrapped != builtin || !PyCFunction_Check(builtin)) goto done;
    blas = PyImport_ImportModule("scipy.linalg.cython_blas");
    if (blas == NULL) goto optional_error;
    api = PyObject_GetAttrString(blas, "__pyx_capi__");
    if (api == NULL) goto optional_error;
    if (!PyDict_CheckExact(api)) goto done;
    capsule = PyDict_GetItemString(api, "ddot"); /* borrowed */
    if (capsule == NULL || !PyCapsule_IsValid(capsule, signature)) goto done;
    state->ddot = (DDotFunction)PyCapsule_GetPointer(capsule, signature);
    if (state->ddot == NULL) goto optional_error;
    state->blas_module = blas;
    Py_INCREF(blas);
    state->ddot_capsule = capsule;
    Py_INCREF(capsule);
    goto done;
optional_error:
    /* An absent/unsupported optional interface keeps ordinary NumPy dot.
       Do not suppress memory exhaustion, interrupts, or unrelated failures. */
    if (PyErr_ExceptionMatches(PyExc_ImportError) ||
            PyErr_ExceptionMatches(PyExc_AttributeError) ||
            PyErr_ExceptionMatches(PyExc_OSError)) PyErr_Clear();
    else success = 0;
done:
    Py_XDECREF(multiarray); Py_XDECREF(canonical); Py_XDECREF(umath);
    Py_XDECREF(dispatcher); Py_XDECREF(builtin); Py_XDECREF(wrapped);
    Py_XDECREF(blas); Py_XDECREF(api);
    return success;
}

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

static int numpy_dot(Functions *functions, PyObject *left, PyObject *right,
                     Py_buffer *left_view, Py_buffer *right_view, double *value) {
    Py_ssize_t length = left_view->len / 8;
    if (functions->ddot != NULL && length <= INT_MAX && length > 0 &&
            right_view->len == left_view->len &&
            Py_TYPE(left) == (PyTypeObject *)functions->ndarray_type &&
            Py_TYPE(right) == (PyTypeObject *)functions->ndarray_type) {
        int n = (int)length, increment = 1;
        /* NumPy owns warning/callback/raise policy. Replay exceptional dots
           through its captured function instead of bypassing that policy. */
        if (feclearexcept(FE_ALL_EXCEPT) == 0) {
            *value = functions->ddot(&n, (double *)left_view->buf, &increment,
                                    (double *)right_view->buf, &increment);
            int flags = fetestexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW);
            if (flags == 0 && isfinite(*value)) return !PyErr_Occurred();
        }
    }
    PyObject *answer = PyObject_CallFunctionObjArgs(functions->dot_function, left, right, NULL);
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
    int tuple_rows = kind == 0 && PyTuple_CheckExact(get_row);
    if ((kind != 0 && kind != 1) || !isfinite(bias) || !isfinite(regularization) ||
            (kind == 1 && regularization <= 0) || (!tuple_rows && !PyCallable_Check(get_row))) {
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
    int ordinary_arrays = (Py_TYPE(current) == (PyTypeObject *)functions->ndarray_type &&
                           Py_TYPE(proposal) == (PyTypeObject *)functions->ndarray_type);
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
        if (tuple_rows) {
            row = PyTuple_GetItem(get_row, (Py_ssize_t)i);
            Py_XINCREF(row);
        } else {
            PyObject *index = PyLong_FromLongLong(i);
            if (index == NULL) goto cleanup;
            row = PyObject_CallFunctionObjArgs(get_row, index, NULL);
            Py_DECREF(index);
        }
        if (row == NULL) {
            if (take_numeric_error(&error)) break;
            goto cleanup;
        }
        if (!vector_buffer(row, &x, 0, 0)) goto cleanup;
        if (x.len != a.len) { PyErr_SetString(PyExc_ValueError, "Sampled vector has wrong length"); goto cleanup; }
        double dot, margin, q, multiplier, new_bias, shrink = 0;
        if (!numpy_dot(functions, current, row, &a, &x, &dot)) {
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
        int check_final_coefficients = 1;
        if (kind == 1) {
            shrink = 1.0 - steps[t] * (2.0 * regularization);
            if (!isfinite(shrink)) { error = PyUnicode_FromString("Nonfinite shrinkage"); goto numerical_end; }
            if (ordinary_arrays) {
                int all_finite = 1;
                for (Py_ssize_t j = 0; j < width; j++) {
                    next[j] = shrink * old[j];
                    all_finite &= (isfinite(next[j]) != 0);
                }
                next[i] -= multiplier;
                if (!all_finite || !isfinite(next[i])) {
                    error = PyUnicode_FromString("Nonfinite coefficients"); goto numerical_end;
                }
                check_final_coefficients = 0;
            } else {
                for (Py_ssize_t j = 0; j < width; j++) next[j] = shrink * old[j];
                next[i] -= multiplier;
            }
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
            if (!numpy_dot(functions, proposal, proposal, &z, &z, &squared_norm)) {
                if (take_numeric_error(&error)) goto numerical_end;
                goto cleanup;
            }
            double norm = sqrt(squared_norm);
            if (scale > 1.0 || norm > 1.0) {
                for (Py_ssize_t j = 0; j < width; j++) next[j] /= norm;
            } else if (ordinary_arrays) {
                /* Plain-array dot cannot mutate the checked, undivided proposal. */
                check_final_coefficients = 0;
            }
        }
        if (check_final_coefficients) for (Py_ssize_t j = 0; j < width; j++) if (!isfinite(next[j])) {
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
        /* Poll only after a committed update and released row. Signal-handler
           exceptions propagate unchanged, including numerical exception types. */
        if ((completed & 1023) == 0 && PyErr_CheckSignals() < 0) goto cleanup;
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
    {"run_steps", run_steps, METH_VARARGS, "Execute ordered scalar updates using guarded BLAS dot and strict arithmetic."},
    {NULL, NULL, 0, NULL}
};
static int module_traverse(PyObject *module, visitproc visit, void *arg) {
    Functions *state = (Functions *)PyModule_GetState(module);
    Py_VISIT(state->dot_function); Py_VISIT(state->copy_function); Py_VISIT(state->empty_function);
    Py_VISIT(state->ndarray_type);
    Py_VISIT(state->blas_module); Py_VISIT(state->ddot_capsule);
    return 0;
}
static int module_clear(PyObject *module) {
    Functions *state = (Functions *)PyModule_GetState(module);
    Py_CLEAR(state->dot_function); Py_CLEAR(state->copy_function); Py_CLEAR(state->empty_function);
    Py_CLEAR(state->ndarray_type);
    state->ddot = NULL;
    Py_CLEAR(state->ddot_capsule); Py_CLEAR(state->blas_module);
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
    state->ndarray_type = PyObject_GetAttrString(numpy, "ndarray");
    Py_DECREF(numpy);
    if (!state->dot_function || !state->copy_function || !state->empty_function || !state->ndarray_type) {
        Py_DECREF(created); return NULL;
    }
    if (!resolve_ddot(state)) { Py_DECREF(created); return NULL; }
    if (PyModule_AddIntConstant(created, "_uses_scipy_ddot", state->ddot != NULL) < 0) {
        Py_DECREF(created); return NULL;
    }
    if (PyModule_AddIntConstant(created, "_supports_tuple_rows", 1) < 0) {
        Py_DECREF(created); return NULL;
    }
    return created;
}
