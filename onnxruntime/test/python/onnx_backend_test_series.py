# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os
import platform
import sys
import unittest

import numpy as np
import onnx
import onnx.backend.test
import onnx.defs

import onnxruntime.backend as c2

pytest_plugins = ("onnx.backend.test.report",)


class OrtBackendTest(onnx.backend.test.BackendTest):
    def __init__(self, backend, parent_module=None):
        super(OrtBackendTest, self).__init__(backend, parent_module)

    @classmethod
    def assert_similar_outputs(cls, ref_outputs, outputs, rtol, atol):
        def assert_similar_array(ref_output, output):
            np.testing.assert_equal(ref_output.dtype, output.dtype)
            if ref_output.dtype == np.object:
                np.testing.assert_array_equal(ref_output, output)
            else:
                np.testing.assert_allclose(ref_output, output, rtol=1e-3, atol=1e-5)

        np.testing.assert_equal(len(ref_outputs), len(outputs))
        for i in range(len(outputs)):
            if isinstance(outputs[i], list):
                for j in range(len(outputs[i])):
                    assert_similar_array(ref_outputs[i][j], outputs[i][j])
            else:
                assert_similar_array(ref_outputs[i], outputs[i])


def apply_filters(filters, category):
    opset_version = f'opset{onnx.defs.onnx_opset_version()}'
    validated_filters = []
    for f in filters[category]:
        if type(f) is list:
            opset_regex = f[0]
            filter_regex = f[1]
            import re
            opset_match = re.match(opset_regex, opset_version)
            if opset_match is not None:
                validated_filters.append(filter_regex)
        else:
            validated_filters.append(f)
    return validated_filters

def create_backend_test(testname=None):
    backend_test = OrtBackendTest(c2, __name__)

    # Type not supported
    backend_test.exclude(r"(FLOAT16)")

    if testname:
        backend_test.include(testname + ".*")
    else:
        # read filters data
        with open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "testdata",
                "onnx_backend_test_series_filters.jsonc",
            )
        ) as f:
            filters_lines = f.readlines()
        filters_lines = [x.split("//")[0] for x in filters_lines]
        filters = json.loads("\n".join(filters_lines))

        current_failing_tests = apply_filters(filters, "current_failing_tests")

        if platform.architecture()[0] == "32bit":
            current_failing_tests += apply_filters(filters, "current_failing_tests_x86")

        if c2.supports_device("DNNL"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_DNNL")

        if c2.supports_device("NNAPI"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_NNAPI")

        if c2.supports_device("OPENVINO_GPU_FP32") or c2.supports_device("OPENVINO_GPU_FP16"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_OPENVINO_GPU")

        if c2.supports_device("OPENVINO_MYRIAD"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_OPENVINO_GPU")
            current_failing_tests += apply_filters(filters, "current_failing_tests_OPENVINO_MYRIAD")

        if c2.supports_device("OPENVINO_CPU_FP32"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_OPENVINO_CPU_FP32")

        if c2.supports_device("MIGRAPHX"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_MIGRAPHX")

        # Skip these tests for a "pure" DML onnxruntime python wheel. We keep these tests enabled for instances where both DML and CUDA
        # EPs are available (Windows GPU CI pipeline has this config) - these test will pass because CUDA has higher precendence than DML
        # and the nodes are assigned to only the CUDA EP (which supports these tests)
        if c2.supports_device("DML") and not c2.supports_device("GPU"):
            current_failing_tests += apply_filters(filters, "current_failing_tests_pure_DML")

        filters = (
            current_failing_tests
            + apply_filters(filters, "tests_with_pre_opset7_dependencies")
            + apply_filters(filters, "unsupported_usages")
            + apply_filters(filters, "failing_permanently")
            + apply_filters(filters, "test_with_types_disabled_due_to_binary_size_concerns")
        )

        backend_test.exclude("(" + "|".join(filters) + ")")
        print("excluded tests:", filters)

        # exclude TRT EP temporarily and only test CUDA EP to retain previous behavior
        os.environ["ORT_ONNX_BACKEND_EXCLUDE_PROVIDERS"] = "TensorrtExecutionProvider"

    # import all test cases at global scope to make
    # them visible to python.unittest.
    globals().update(backend_test.enable_report().test_cases)

    return backend_test


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Run the ONNX backend tests using ONNXRuntime.",
    )

    # Add an argument to match a single test name, by adding the name to the 'include' filter.
    # Using -k with python unittest (https://docs.python.org/3/library/unittest.html#command-line-options)
    # doesn't work as it filters on the test method name (Runner._add_model_test) rather than inidividual
    # test case names.
    parser.add_argument(
        "-t",
        "--test-name",
        dest="testname",
        type=str,
        help="Only run tests that match this value. Matching is regex based, and '.*' is automatically appended",
    )

    # parse just our args. python unittest has its own args and arg parsing, and that runs inside unittest.main()
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    return args


if __name__ == "__main__":
    args = parse_args()

    backend_test = create_backend_test(args.testname)
    unittest.main()
