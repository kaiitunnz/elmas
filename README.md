# Installer trace_processor

## Install GLIBC_2.34

1. Follow the instruction from [here](https://askubuntu.com/questions/1345342/how-to-install-glibc-2-32-when-i-already-have-glibc2-31).
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

## Configure trace_processor

1. Install `patchelf`.
    ```bash
    pip install patchelf
    ```
2. Patch the executable.
    ```bash
    patchelf --set-interpreter $HOME/glibc/glibc-2.34-install/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/glibc/glibc-2.34-install ~/.local/share/perfetto/prebuilts/trace_processor_shell
    ```