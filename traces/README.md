# Profiling

This directory is used to store vLLM's profiling traces. See [scripts/profiling.py](../scripts/profiling.py) for the client script to profile vLLM and [src/](../src/agents/utils/vllm/start_server.py) for the script to run vLLM in profiling mode.

## Getting Started

### Install Perfetto's trace_processor

1. Run the following commands to download Perfetto's Trace Processor

   ```bash
   curl -LO https://get.perfetto.dev/trace_processor
   chmod +x ./trace_processor
   ```

2. Install Perfetto's Python library.

   ```bash
   pip install perfetto
   ```

3. Use Perfetto in a Python program. For example,

   ```python
   from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig

   trace_file = "/path/to/trace/file"
   trace_processor_bin = "/path/to/trace_processor"

   tp = TraceProcessor(
       trace=trace_file,
       config=TraceProcessorConfig(bin_path=trace_processor_bin),
   )
   qr_it = tp.query('SELECT ts, dur, name FROM slice')
   for row in qr_it:
       print(row.ts, row.dur, row.name)
   ```

In case you encounter an error about an incompatible glibc version, follow the instructions in the next section.

### Install a compatible glib version locally

#### 1. Install GLIBC_2.34.

1. Run the following commands to install glibc locally ([ref.](https://askubuntu.com/questions/1345342/how-to-install-glibc-2-32-when-i-already-have-glibc2-31)).
   ```bash
   mkdir $HOME/glibc/ && cd $HOME/glibc
   wget http://ftp.gnu.org/gnu/libc/glibc-2.34.tar.gz
   tar -xvzf glibc-2.34.tar.gz
   mkdir build
   mkdir glibc-2.34-install
   cd build
   ~/glibc/glibc-2.34/configure --prefix=$HOME/glibc/glibc-2.34-install
   make
   make install
   ```
2. Copy the system's `libgcc_s.so.1` to your newly installed glibc.
   ```bash
   cp /lib/x86_64-linux-gnu/libgcc_s.so.1 ~/glibc/glibc-2.34-install/lib
   ```

#### 2. Patch trace_processor.

1. Install `patchelf`.
   ```bash
   pip install patchelf
   ```
2. Patch the executable.
   ```bash
   patchelf --set-interpreter $HOME/glibc/glibc-2.34-install/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/glibc/glibc-2.34-install ~/.local/share/perfetto/prebuilts/trace_processor_shell
   ```
