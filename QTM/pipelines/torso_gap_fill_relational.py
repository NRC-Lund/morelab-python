from .gap_fill_relational import add_marker_prefix, gap_fill_relational


base_marker_names = [
    "SpineThoracic2",
    "SpineThoracic12",
    "Chest",
    "ChestLow",
    "LShoulderTop",
    "RShoulderTop",
]

base_gap_fill_rules = {
    "SpineThoracic2": [
        ["SpineThoracic12", "Chest", "ChestLow"],
        ["Chest", "SpineThoracic12", "LShoulderTop"],
        ["Chest", "SpineThoracic12", "RShoulderTop"],
        ["SpineThoracic12", "LShoulderTop", "RShoulderTop"],
    ],

    "SpineThoracic12": [
        ["SpineThoracic2", "ChestLow", "Chest"],
        ["ChestLow", "SpineThoracic2", "LShoulderTop"],
        ["ChestLow", "SpineThoracic2", "RShoulderTop"],
        ["SpineThoracic2", "LShoulderTop", "RShoulderTop"],
    ],

    "Chest": [
        ["ChestLow", "SpineThoracic2", "SpineThoracic12"],
        ["SpineThoracic2", "LShoulderTop", "RShoulderTop"],
        ["ChestLow", "LShoulderTop", "RShoulderTop"],
        ["SpineThoracic12", "LShoulderTop", "RShoulderTop"],
    ],

    "ChestLow": [
        ["Chest", "SpineThoracic12", "SpineThoracic2"],
        ["SpineThoracic12", "LShoulderTop", "RShoulderTop"],
        ["Chest", "LShoulderTop", "RShoulderTop"],
        ["SpineThoracic2", "LShoulderTop", "RShoulderTop"],
    ],

    "LShoulderTop": [
        ["RShoulderTop", "SpineThoracic2", "SpineThoracic12"],
        ["SpineThoracic2", "Chest", "ChestLow"],
        ["Chest", "SpineThoracic2", "SpineThoracic12"],
        ["ChestLow", "SpineThoracic12", "SpineThoracic2"],
    ],

    "RShoulderTop": [
        ["LShoulderTop", "SpineThoracic2", "SpineThoracic12"],
        ["SpineThoracic2", "Chest", "ChestLow"],
        ["Chest", "SpineThoracic2", "SpineThoracic12"],
        ["ChestLow", "SpineThoracic12", "SpineThoracic2"],
    ],
}

marker_names, gap_fill_rules = add_marker_prefix(
    base_marker_names, base_gap_fill_rules, "Q_")


def torso_gap_fill_relational():
    gap_fill_relational(marker_names, gap_fill_rules)
