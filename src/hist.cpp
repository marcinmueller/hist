#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <immintrin.h>

using TYPE = double;

static PyObject *
hist(PyObject *module, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2)
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires 2 positional arguments: array and number of bins");
        return NULL;
    }

    if (!PyArray_Check(args[0]) || PyArray_NDIM((PyArrayObject*)args[0]) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires a 1-dimensional numpy array object");
        return NULL;
    }
    
    size_t nSize = PyArray_DIM((PyArrayObject*)args[0], 0);
    if (nSize == 0)
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires a non-empty numpy array object");
        return NULL;
    }
    
    auto data = static_cast<const TYPE*>(PyArray_DATA((PyArrayObject*)args[0]));

    if (!PyLong_Check(args[1]))
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires second positional argument to be of type int");
        return NULL;
    }
    
    size_t nBucketCount = PyLong_AsSize_t(args[1]);
    if (nBucketCount < 1)
    {
        PyErr_SetString(PyExc_TypeError, "number of buckets must be at least 1");
        return NULL;
    }

    TYPE dMin, dMax;
    dMin = dMax = data[0];
    for (size_t i = 1; i < nSize; ++i)
    {
        if (data[i] < dMin)
            dMin = data[i];
        if (data[i] > dMax)
            dMax = data[i];
    }

    auto dBucketWidth = (dMax-dMin)/nBucketCount;
    if (dBucketWidth == 0)
    {
        PyErr_SetString(PyExc_TypeError, "data array contains same values");
        return NULL;        
    }

    // Allocate one more bucket to avoid crashes
    auto buckets = new uint64_t[nBucketCount+1];
    memset(buckets, 0, sizeof(uint64_t)*(nBucketCount+1));
    for (size_t i = 0; i < nSize; ++i)
        ++buckets[static_cast<size_t>((data[i] - dMin)/dBucketWidth)];
    // Add the overflow counts (where data was exactly max of data) to the last bucket
    buckets[nBucketCount-1] += buckets[nBucketCount];

    npy_intp dims[] = { static_cast<npy_intp>(nBucketCount) };
    PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_UINT64, buckets);
    PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);
    return arr;
}

static PyObject *
hist2(PyObject *module, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2)
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires 2 positional arguments: array and number of bins");
        return NULL;
    }

    if (!PyArray_Check(args[0]) || PyArray_NDIM((PyArrayObject*)args[0]) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires a 1-dimensional numpy array object");
        return NULL;
    }
    
    size_t nSize = PyArray_DIM((PyArrayObject*)args[0], 0);
    if (nSize == 0)
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires a non-empty numpy array object");
        return NULL;
    }
    
    auto data = static_cast<const TYPE*>(PyArray_DATA((PyArrayObject*)args[0]));

    if (!PyLong_Check(args[1]))
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires second positional argument to be of type int");
        return NULL;
    }
    
    size_t nBucketCount = PyLong_AsSize_t(args[1]);
    if (nBucketCount < 1)
    {
        PyErr_SetString(PyExc_TypeError, "number of buckets must be at least 1");
        return NULL;
    }

    TYPE dMin, dMax;
    dMin = dMax = data[0];
    for (size_t i = 1; i < nSize; ++i)
    {
        if (data[i] < dMin)
            dMin = data[i];
        if (data[i] > dMax)
            dMax = data[i];
    }

    auto dBucketWidth = (dMax-dMin)/nBucketCount;
    if (dBucketWidth == 0)
    {
        PyErr_SetString(PyExc_TypeError, "data array contains same values");
        return NULL;        
    }

    // Allocate one more bucket to avoid crashes
    auto buckets1 = new uint64_t[nBucketCount+1];
    auto buckets2 = new uint64_t[nBucketCount+1];
    memset(buckets1, 0, sizeof(uint64_t)*(nBucketCount+1));
    memset(buckets2, 0, sizeof(uint64_t)*(nBucketCount+1));
    
    size_t nAlignedSize = nSize - nSize%2;
    for (size_t i = 0; i < nAlignedSize; i += 2)
    {
        ++buckets1[static_cast<size_t>((data[i] - dMin)/dBucketWidth)];
        ++buckets2[static_cast<size_t>((data[i+1] - dMin)/dBucketWidth)];
    }
    for (size_t i = nAlignedSize; i < nSize; ++i)
        ++buckets1[static_cast<size_t>((data[i] - dMin)/dBucketWidth)];

    // Merge both histograms
    for (size_t i = 0; i < nBucketCount + 1; ++i)
        buckets1[i] += buckets2[i];
    delete [] buckets2;

    // Add the overflow counts (where data was exactly max of data) to the last bucket
    buckets1[nBucketCount-1] += buckets1[nBucketCount];

    npy_intp dims[] = { static_cast<npy_intp>(nBucketCount) };
    PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_UINT64, buckets1);
    PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);
    return arr;
}

static PyObject *
hist4(PyObject *module, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2)
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires 2 positional arguments: array and number of bins");
        return NULL;
    }

    if (!PyArray_Check(args[0]) || PyArray_NDIM((PyArrayObject*)args[0]) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires a 1-dimensional numpy array object");
        return NULL;
    }
    
    size_t nSize = PyArray_DIM((PyArrayObject*)args[0], 0);
    if (nSize == 0)
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires a non-empty numpy array object");
        return NULL;
    }
    
    auto data = static_cast<const TYPE*>(PyArray_DATA((PyArrayObject*)args[0]));

    if (!PyLong_Check(args[1]))
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires second positional argument to be of type int");
        return NULL;
    }
    
    size_t nBucketCount = PyLong_AsSize_t(args[1]);
    if (nBucketCount < 1)
    {
        PyErr_SetString(PyExc_TypeError, "number of buckets must be at least 1");
        return NULL;
    }

    TYPE dMin, dMax;
    dMin = dMax = data[0];
    for (size_t i = 1; i < nSize; ++i)
    {
        if (data[i] < dMin)
            dMin = data[i];
        if (data[i] > dMax)
            dMax = data[i];
    }

    auto dBucketWidth = (dMax-dMin)/nBucketCount;
    if (dBucketWidth == 0)
    {
        PyErr_SetString(PyExc_TypeError, "data array contains same values");
        return NULL;        
    }

    // Allocate one more bucket to avoid crashes
    auto buckets1 = new uint64_t[nBucketCount+1];
    auto buckets2 = new uint64_t[nBucketCount+1];
    auto buckets3 = new uint64_t[nBucketCount+1];
    auto buckets4 = new uint64_t[nBucketCount+1];
    memset(buckets1, 0, sizeof(uint64_t)*(nBucketCount+1));
    memset(buckets2, 0, sizeof(uint64_t)*(nBucketCount+1));
    memset(buckets3, 0, sizeof(uint64_t)*(nBucketCount+1));
    memset(buckets4, 0, sizeof(uint64_t)*(nBucketCount+1));
    
    size_t nAlignedSize = nSize - nSize%4;
    for (size_t i = 0; i < nAlignedSize; i += 4)
    {
        ++buckets1[static_cast<size_t>((data[i] - dMin)/dBucketWidth)];
        ++buckets2[static_cast<size_t>((data[i+1] - dMin)/dBucketWidth)];
        ++buckets3[static_cast<size_t>((data[i+2] - dMin)/dBucketWidth)];
        ++buckets4[static_cast<size_t>((data[i+3] - dMin)/dBucketWidth)];
    }
    for (size_t i = nAlignedSize; i < nSize; ++i)
        ++buckets1[static_cast<size_t>((data[i] - dMin)/dBucketWidth)];

    // Merge the histograms
    for (size_t i = 0; i < nBucketCount + 1; ++i)
        buckets1[i] += buckets2[i] + buckets3[i] + buckets4[i];
    delete [] buckets2;
    delete [] buckets3;
    delete [] buckets4;

    // Add the overflow counts (where data was exactly max of data) to the last bucket
    buckets1[nBucketCount-1] += buckets1[nBucketCount];

    npy_intp dims[] = { static_cast<npy_intp>(nBucketCount) };
    PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_UINT64, buckets1);
    PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);
    return arr;
}

static PyObject *
hist_avx2(PyObject *module, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2)
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires 2 positional arguments: array and number of bins");
        return NULL;
    }

    if (!PyArray_Check(args[0]) || PyArray_NDIM((PyArrayObject*)args[0]) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires a 1-dimensional numpy array object");
        return NULL;
    }
    
    size_t nSize = PyArray_DIM((PyArrayObject*)args[0], 0);
    if (nSize == 0)
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires a non-empty numpy array object");
        return NULL;
    }
    
    auto data = static_cast<const TYPE*>(PyArray_DATA((PyArrayObject*)args[0]));

    if (!PyLong_Check(args[1]))
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires second positional argument to be of type int");
        return NULL;
    }
    
    size_t nBucketCount = PyLong_AsSize_t(args[1]);
    if (nBucketCount < 1)
    {
        PyErr_SetString(PyExc_TypeError, "number of buckets must be at least 1");
        return NULL;
    }

    TYPE dMin, dMax;
    dMin = dMax = data[0];
    for (size_t i = 1; i < nSize; ++i)
    {
        if (data[i] < dMin)
            dMin = data[i];
        if (data[i] > dMax)
            dMax = data[i];
    }

    auto dBucketWidth = (dMax-dMin)/nBucketCount;
    if (dBucketWidth == 0)
    {
        PyErr_SetString(PyExc_TypeError, "data array contains same values");
        return NULL;        
    }

    // Allocate one more bucket to avoid crashes
    auto buckets = new uint64_t[nBucketCount+1];
    memset(buckets, 0, sizeof(uint64_t)*(nBucketCount+1));

    __m256d minv = _mm256_set1_pd(dMin);
    __m256d widthv = _mm256_set1_pd(dBucketWidth);
    for (size_t i = 0; i < nSize/4; ++i)
    {
        __m256d datav = _mm256_loadu_pd(&data[4*i]);
        datav = _mm256_sub_pd(datav, minv);
        datav = _mm256_div_pd(datav, widthv);
        __m128i iv = _mm256_cvttpd_epi32(datav);
        
        ++buckets[_mm_extract_epi32(iv, 0)];
        ++buckets[_mm_extract_epi32(iv, 1)];
        ++buckets[_mm_extract_epi32(iv, 2)];
        ++buckets[_mm_extract_epi32(iv, 3)];
    }

    for (size_t i = (nSize/4)*4; i < nSize; ++i)
        ++buckets[static_cast<size_t>((data[i] - dMin)/dBucketWidth)];
    
    // Add the overflow counts (where data was exactly max of data) to the last bucket
    buckets[nBucketCount-1] += buckets[nBucketCount];

    npy_intp dims[] = { static_cast<npy_intp>(nBucketCount) };
    PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_UINT64, buckets);
    PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);
    return arr;
}

static PyObject *
hist_avx512(PyObject *module, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2)
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires 2 positional arguments: array and number of bins");
        return NULL;
    }

    if (!PyArray_Check(args[0]) || PyArray_NDIM((PyArrayObject*)args[0]) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires a 1-dimensional numpy array object");
        return NULL;
    }
    
    size_t nSize = PyArray_DIM((PyArrayObject*)args[0], 0);
    if (nSize == 0)
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires a non-empty numpy array object");
        return NULL;
    }
    
    auto data = static_cast<const TYPE*>(PyArray_DATA((PyArrayObject*)args[0]));

    if (!PyLong_Check(args[1]))
    {
        PyErr_SetString(PyExc_TypeError, "hist() requires second positional argument to be of type int");
        return NULL;
    }
    
    size_t nBucketCount = PyLong_AsSize_t(args[1]);
    if (nBucketCount < 1)
    {
        PyErr_SetString(PyExc_TypeError, "number of buckets must be at least 1");
        return NULL;
    }

    TYPE dMin, dMax;
    dMin = dMax = data[0];
    for (size_t i = 1; i < nSize; ++i)
    {
        if (data[i] < dMin)
            dMin = data[i];
        if (data[i] > dMax)
            dMax = data[i];
    }

    auto dBucketWidth = (dMax-dMin)/nBucketCount;
    if (dBucketWidth == 0)
    {
        PyErr_SetString(PyExc_TypeError, "data array contains same values");
        return NULL;        
    }

    // Allocate one more bucket to avoid crashes
    auto buckets = new uint64_t[nBucketCount+1];
    memset(buckets, 0, sizeof(uint64_t)*(nBucketCount+1));

    __m512d minv = _mm512_set1_pd(dMin);
    __m512d widthv = _mm512_set1_pd(dBucketWidth);
    for (size_t i = 0; i < nSize/8; ++i)
    {
        __m512d datav = _mm512_loadu_pd(&data[8*i]);
        datav = _mm512_sub_pd(datav, minv);
        datav = _mm512_div_pd(datav, widthv);
        __m256i iv = _mm512_cvttpd_epi32(datav);
        
        ++buckets[_mm256_extract_epi32(iv, 0)];
        ++buckets[_mm256_extract_epi32(iv, 1)];
        ++buckets[_mm256_extract_epi32(iv, 2)];
        ++buckets[_mm256_extract_epi32(iv, 3)];
        ++buckets[_mm256_extract_epi32(iv, 4)];
        ++buckets[_mm256_extract_epi32(iv, 5)];
        ++buckets[_mm256_extract_epi32(iv, 6)];
        ++buckets[_mm256_extract_epi32(iv, 7)];
    }

    for (size_t i = (nSize/8)*8; i < nSize; ++i)
        ++buckets[static_cast<size_t>((data[i] - dMin)/dBucketWidth)];
    
    // Add the overflow counts (where data was exactly max of data) to the last bucket
    buckets[nBucketCount-1] += buckets[nBucketCount];

    npy_intp dims[] = { static_cast<npy_intp>(nBucketCount) };
    PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_UINT64, buckets);
    PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);
    return arr;
}

static PyMethodDef hist_methods[] = {
    {"hist",  (PyCFunction)(void(*)(void))hist, METH_FASTCALL, "Compute histograph of double array"},
    {"hist2",  (PyCFunction)(void(*)(void))hist2, METH_FASTCALL, "Compute histograph of double array"},
    {"hist4",  (PyCFunction)(void(*)(void))hist4, METH_FASTCALL, "Compute histograph of double array"},
    {"hist_avx2",  (PyCFunction)(void(*)(void))hist_avx2, METH_FASTCALL, "Compute histograph of double array (AVX2)"},
    {"hist_avx512",  (PyCFunction)(void(*)(void))hist_avx512, METH_FASTCALL, "Compute histograph of double array (AVX512)"},
    {NULL, NULL, 0, NULL}       /* Sentinel */
};

static struct PyModuleDef hist_module = {
    PyModuleDef_HEAD_INIT,
    "hist",     /* name of module */
    NULL,       /* module documentation, may be NULL */
    -1,         /* size of per-interpreter state of the module,
                    or -1 if the module keeps state in global variables. */
    hist_methods
};

PyMODINIT_FUNC
PyInit_hist(void)
{
    PyObject* module = PyModule_Create(&hist_module);
    if (!module)
        return NULL;
    import_array();
    if (PyErr_Occurred())
        return NULL;
    return module;
}