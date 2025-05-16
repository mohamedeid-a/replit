{ pkgs }: {
    deps = [
        pkgs.python310Full
        pkgs.python310Packages.pip
        pkgs.python310Packages.flask
        pkgs.python310Packages.numpy
        pkgs.python310Packages.pandas
        pkgs.python310Packages.scikitlearn
        pkgs.python310Packages.joblib
        pkgs.python310Packages.requests
        pkgs.python310Packages.transformers
    ];
    env = {
        PYTHONPATH = "${pkgs.python310Full}/bin/python3.10";
        LANG = "en_US.UTF-8";
        PIP_ROOT_USER_ACTION = "ignore";
    };
}
