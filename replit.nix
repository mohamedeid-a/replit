{ pkgs }: {
    deps = [
        pkgs.python38Full
    ];
    env = {
        PYTHONHOME = "${pkgs.python38Full}";
        PYTHONBIN = "${pkgs.python38Full}/bin/python3.8";
        LANG = "en_US.UTF-8";
        STDERRED_PATH = "${pkgs.stderred}/lib/libstderred.so";
        PYTHONPATH = ".";
    };
}
