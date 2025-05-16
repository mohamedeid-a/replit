{ pkgs }: {
    deps = [
        pkgs.python3
        pkgs.python3Packages.flask
        pkgs.python3Packages.numpy
        pkgs.python3Packages.pandas
        pkgs.python3Packages.scikit-learn
    ];
}
