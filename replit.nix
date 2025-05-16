{ pkgs }: {
    deps = [
        pkgs.python39
        pkgs.python39Packages.pip
        pkgs.python39Packages.flask
        pkgs.python39Packages.numpy
        pkgs.python39Packages.pandas
        pkgs.python39Packages.scikitlearn
        pkgs.python39Packages.requests
        pkgs.python39Packages.transformers
        pkgs.python39Packages.torch
    ];
}
