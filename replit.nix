{ pkgs }: {
    deps = [
        pkgs.python310
    ];
    env = {
        PYTHONPATH = "${pkgs.python310}/bin/python3.10";
        PIP_ROOT_USER_ACTION = "ignore";
    };
}
