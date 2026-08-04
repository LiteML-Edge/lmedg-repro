# Environment and Build Reports

> **Navigation:** For the complete manuscript-review sequence, begin with [`REVIEWER_START_HERE.md`](../REVIEWER_START_HERE.md). This directory is limited to compilation records and captured environment evidence.


This directory is the reviewer-facing index for the retained build records and the captured software and host environment.

## Reviewer access

Start with this file, then open the relevant subdirectory:

- [`final_builds/`](final_builds/README.md): six final REPLAY compilation records, covering three models and two hardware targets;
- [`system reports/`](system%20reports/README.md): Windows, Python, PlatformIO, platform, toolchain, package, and project-local library reports.

The directory name `system reports` contains a space. Quote the path in command-line operations, for example:

```powershell
Get-ChildItem ".\environment_reports\system reports"
```

## Final REPLAY build records

The six build records correspond to the REPLAY firmware configurations used for stage-wise conformance verification. FIELD configurations use the same PlatformIO projects and build procedure, with the acquisition mode selected by a compile-time macro.

These records provide final compilation evidence for MLP, Conv1D Tiny, and LSTM on WEMOS LOLIN32 and NUCLEO-F411RE. They are build logs rather than firmware binaries. See [`final_builds/README.md`](final_builds/README.md) for the exact file mapping, interpretation, encoding, and reproduction commands.

## System reports

The `system reports/` directory records the experiment workstation and software environment used to generate and execute the retained artifacts. It includes the Windows host, Python virtual environment, PlatformIO Core, target platforms, frameworks, tools, toolchains, managed packages, and project-local libraries. See [`system reports/README.md`](system%20reports/README.md) for the exact role of each file.

Absolute workstation paths retained inside captured reports identify the environment at capture time. All reproduction instructions in the pack use repository-relative paths.
