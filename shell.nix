{
  mkShell,
  pkgs,
  ...
}:
mkShell {
  packages = with pkgs; [
    pyright
    clang-tools
    black
    ruff
    (python3.withPackages (
      ps: with ps; [
        torch
        numpy
        ipython
        pytest
        pip
      ]
    ))
  ];

  shellHook = ''
    rm -rf .venv
    mkdir -p .venv/bin
    ln -sfT "$(which python3)" .venv/bin/python3
    ln -sfT "$(which python3)" .venv/bin/python
  '';
}
