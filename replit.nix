{ pkgs }: {
    deps = [
        pkgs.python3
        pkgs.python3Packages.pip
        pkgs.gcc
    ];
    env = {
        PYTHONPATH = "${pkgs.python3}/bin/python3";
        LANG = "en_US.UTF-8";
        PIP_ROOT_USER_ACTION = "ignore";
    };
}
