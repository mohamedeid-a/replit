{ pkgs }: {
    deps = with pkgs; [
        python310
        python310Packages.flask
        python310Packages.numpy
        python310Packages.pandas
        python310Packages.scikit-learn
        python310Packages.requests
        python310Packages.pip
        gcc
    ];
}
