{
  mkShell,
  pkgs,
  ...
}:
mkShell {
  packages = with pkgs; [
    gcc
    gdb
    cmake
    rocmPackages.clr
    rocmPackages.rocminfo
    black
    ruff
    (python3.withPackages (
      ps: with ps; [
        torchWithCuda
        numpy
        ipython
        pytest
        pip
      ]
    ))
  ];

  shellHook = ''
    export CUDA_HOME="${pkgs.cudatoolkit}"
    export ROCM_HOME="${pkgs.rocmPackages.clr}"
    export HIP_PATH="$ROCM_HOME"
    export PATH="$CUDA_HOME/bin:$ROCM_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$ROCM_HOME/lib:$LD_LIBRARY_PATH"
    export CXX=hipcc
  '';
}
