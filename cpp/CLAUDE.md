# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About libcudf

libcudf is a CUDA C++ library providing GPU-accelerated data-parallel algorithms for processing column-oriented tabular data. It is part of the RAPIDS ecosystem and provides Apache Arrow-compatible data structures.

## Build Commands

### Configure
```bash
configure-cudf-cpp -DCMAKE_CUDA_ARCHITECTURES=native # C++ library and tests
configure-cudf-cpp -DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_BENCHMARKS=ON # C++ library, tests and benchmarks

```
### Basic Build
```bash
build-cudf-cpp
```

### Clean
```bash
clean-cudf-cpp
```

## Testing

### Running C++ Tests
```bash
# Run single test executable directly
cpp/build/gtests/<TEST_NAME>
```

### Running Python Tests
```bash
cd python
pytest -v cudf/cudf/tests
pytest -v dask_cudf/dask_cudf/
```

## Benchmarks

Benchmarks are **not built by default**. Enable with:
```bash
configure-cudf-cpp -DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_BENCHMARKS=ON # C++ library, tests and benchmarks
build-cudf-cpp
```

Run benchmarks:
```bash
# List available benchmarks
ls cpp/build/latest/benchmarks/*_NVBENCH

# Run a specific benchmark
./cpp/build/latest/benchmarks/<NAME>_NVBENCH

# View benchmark options
./cpp/build/latest/benchmarks/<NAME>_NVBENCH --help
```

## Architecture Overview

### Core Data Structures

**Ownership Model**: libcudf uses a view/ownership pattern where:
- **Owning types** (`column`, `table`): Own device memory, use RAII
- **View types** (`column_view`, `table_view`): Non-owning, lightweight, zero-copy access
- **Device view types** (`column_device_view`): Trivially copyable views usable in CUDA kernels

**Key Types**:
- `cudf::column` / `cudf::column_view`: Single array of typed data with optional null mask
- `cudf::table` / `cudf::table_view`: Collection of columns with equal row count
- `cudf::size_type`: Signed 32-bit integer for sizes/indices (max 2,147,483,647)
- `cudf::scalar`: Single element of a data type

### Directory Structure

```
cpp/
├── include/cudf/          # Public API headers (.hpp)
│   ├── <feature>.hpp      # Public APIs (e.g., copying.hpp, join.hpp)
│   └── detail/            # Internal APIs used across translation units
├── src/                   # Implementation files
│   ├── <feature>/         # Implementation for corresponding public header
│   │   ├── *.cu           # CUDA implementation files
│   │   └── *.cpp          # C++ implementation files
│   └── jit/               # JIT compilation infrastructure (Jitify2)
├── tests/                 # Unit tests (Google Test)
│   └── <feature>/         # Tests matching src structure
│       └── *_tests.cu/cpp
└── benchmarks/            # Performance benchmarks (NVBench)
    └── <feature>/
        └── *.cu/cpp
```

**Naming Convention**: The public header name matches the implementation directory.
- Header: `include/cudf/copying.hpp`
- Implementation: `src/copying/*.cu/cpp`
- Tests: `tests/copying/*_tests.cu/cpp`

### Memory Management

All device memory allocation uses **RMM** (RAPIDS Memory Manager):
- Default memory resource: `cudf::get_current_device_resource_ref()`
- Memory resource parameters: `rmm::device_async_resource_ref mr`
- All public APIs should default `mr` parameter to `cudf::get_current_device_resource_ref()`

### Stream Management

CUDA streams are passed via `rmm::cuda_stream_view`:
- Default stream: `cudf::get_default_stream()`
- All public APIs should default `stream` parameter to `cudf::get_default_stream()`

### JIT Compilation

libcudf uses Jitify2 for runtime CUDA kernel compilation:
- JIT infrastructure: `cpp/src/jit/`
- Accessors for JIT: `cpp/src/jit/accessors.cuh`
- Cache: `JITIFY_USE_CACHE` option (ON by default)
- Used for operations like binary ops, transforms, and AST evaluation

## Code Style Guidelines

### Naming Conventions
- **snake_case** for functions, variables, classes
- **PascalCase** for template parameters and test names
- Private member variables: prefix with underscore (`_member`)
- No Hungarian notation (except sometimes for device/host pairs)

### File Extensions
- `.hpp`: C++ headers
- `.cpp`: C++ source files
- `.cu`: CUDA C++ source files (use only when necessary)
- `.cuh`: Headers with CUDA device code (use only when necessary)

**Rule**: Only use `.cu`/`.cuh` if file contains:
- `__device__`, `__global__`, or other nvcc-only symbols
- Thrust algorithms with device execution policy
- `thrust::device_vector` (prefer `rmm::device_uvector` in tests)

### Code Formatting
- Enforced via `clang-format` (see `cpp/.clang-format`)
- Run `pre-commit run` before committing
- Install pre-commit hooks: `pre-commit install`
- Prefer "east const": `type const` not `const type`

### Include Organization
Group includes from "nearest" to "farthest":
1. Local internal headers (use quotes: `"internal.hpp"`)
2. cuDF public headers (use brackets: `<cudf/column.hpp>`)
3. RAPIDS libraries (RMM, nvtext)
4. CUDA libraries (Thrust, CUB, libcu++)
5. Third-party dependencies
6. Standard library headers

Use `#pragma once` for all header guards.

### API Conventions

**Public APIs**:
- Input: Take `*_view` types (e.g., `column_view`, `table_view`)
- Output: Return `std::unique_ptr` to owning types
- Mark namespace with `CUDF_EXPORT`: `namespace CUDF_EXPORT cudf { ... }`
- Example: `std::unique_ptr<table> sort(table_view const& input);`

**Internal/Detail APIs**:
- Place in `cudf::detail` namespace
- Headers go in `include/cudf/detail/` or `include/cudf/<feature>/detail/`
- May take `mutable_*_view` types for in-place operations

## Testing Guidelines

### Test Structure
- Inherit from `cudf::test::BaseFixture` (ensures RMM initialization)
- Use typed tests for type-generic algorithms
- Write test code in global namespace (not `cudf::` or `cudf::test::`)
- Include `<cudf_test/cudf_gtest.hpp>` instead of `gtest/gtest.h`

### What to Test
- Empty input handling (usually produces empty output)
- Boundary conditions: < 32, = 32, > 32, > 64 rows (for bitmask operations)
- Multi-block launches (large enough data to require multiple blocks)
- Sliced columns (non-zero offset)
- Null values and validity masks
- String tests: Include non-ASCII UTF-8 characters
- Lists/Strings: Test empty vs null vs elements with nulls

### Test File Naming
- Prefer `.cpp` over `.cu` (nvcc slower for host code)
- Use `rmm::device_uvector` and `column_wrapper` types (work in `.cpp`)
- Avoid `thrust::device_vector` in tests (requires `.cu`)

## Common Patterns

### Creating Columns
```cpp
// From existing data
auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                      num_rows,
                                      cudf::mask_state::UNALLOCATED,
                                      stream,
                                      mr);

// Get mutable view for kernel writes
auto mutable_view = col->mutable_view();
```

### Launching Kernels
```cpp
cudf::detail::grid_1d grid{num_elements, block_size};
my_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(args...);
```

### Working with Null Masks
```cpp
// Check if column has nulls
if (col.nullable()) {
  auto null_count = col.null_count();
}

// Access null mask
auto mask = col.null_mask();
```

## Development Workflow

1. **Find/Create Issue**: Check for "good first issue" or "help wanted" labels
2. **Create Branch**: Name descriptively (e.g., `fix-join-null-handling`)
3. **Code**: Follow style guidelines, write tests and benchmarks
4. **Format**: Run `pre-commit run` or install hooks
5. **Test**: Run relevant tests via `ctest`
6. **Commit**: Use descriptive messages
7. **PR**: Open against appropriate base branch (usually `branch-XX.YY`)
8. **CI**: Ensure all checks pass (C++ changes need 2 approvals from cudf-cpp-codeowners)

## Additional Resources

- [C++ Developer Guide](cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md)
- [Documentation Guide](cpp/doxygen/developer_guide/DOCUMENTATION.md)
- [Testing Guide](cpp/doxygen/developer_guide/TESTING.md)
- [Benchmarking Guide](cpp/doxygen/developer_guide/BENCHMARKING.md)
- [Profiling Guide](cpp/doxygen/developer_guide/PROFILING.md)
- [RAPIDS Documentation](https://docs.rapids.ai/)
