# Upstream inputs

This artifact's Apache-2.0 code license does not relicense upstream datasets.

- Spider 2.0-DBT: https://github.com/xlang-ai/Spider2 , pinned at cafb867313aab4e674652054198f383cf4018943. The reproduction uses project-local SQL references under spider2-dbt/examples. Obtain upstream files through the documented checkout and consult the upstream repository's license and dataset terms.
- DW-Bench: https://github.com/AJamal27891/dw-bench , immutable v1.0.0 release archive. SHA256: 94a4057581467745abba3953add69bd6cc4e9e984b05e7894e9f2238ff9674f9. The fetcher records the input identity and file locations. Consult the release's license and underlying dataset terms.

The JSON files in expected/ are derived experiment outputs, including project/model labels needed to audit the results. Raw upstream SQL projects, graph archives, and gold question files are not bundled. Python package dependencies are installed separately and retain their own licenses.

Both upstream distributions include MIT licenses. Verbatim notices are retained under third_party/. This records the shipped notices; consult upstream for any component-specific terms.
