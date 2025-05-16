{ pkgs }: {
    deps = [
        pkgs.python38
        pkgs.python38Packages.pip
        pkgs.python38Packages.flask
        pkgs.python38Packages.numpy
        pkgs.python38Packages.pandas
        pkgs.python38Packages.scikit-learn
    ];
}
