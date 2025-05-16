{ pkgs }: {
    deps = [
        pkgs.python3Full
        pkgs.poetry
        pkgs.gcc
    ];
    env = {
        PYTHONBIN = "${pkgs.python3Full}/bin/python3";
        LANG = "en_US.UTF-8";
    };
}
